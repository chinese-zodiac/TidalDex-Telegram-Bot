import os
import json
import asyncio
from dotenv import load_dotenv
from web3 import Web3
from telegram import Bot, Update, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, ContextTypes, ConversationHandler, MessageHandler, filters
from download_tokens import download_token_list
from openai import OpenAI
import requests
from io import BytesIO
from PIL import Image
import random
import cv2
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Retrieve configuration from environment variables
BSC_RPC_URL = os.getenv("BSC_RPC_URL")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOKENS_FILE = os.getenv("TOKENS_FILE", "tidaldex_tokens.json")
PAIRS_FILE = os.getenv("PAIRS_FILE", "tracked_pairs.json")

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
)

# Constants
WAITING_FOR_IMAGE = 1  # State for conversation handler

# Load ABI from file
with open('pairABI.json', 'r') as f:
    PAIR_ABI = json.load(f)

# Connect to BSC
w3 = Web3(Web3.HTTPProvider(BSC_RPC_URL))

# Global variable to store filters
pair_filters = {}

def load_pairs():
    """Load pairs data with chat-specific tracking"""
    try:
        with open(PAIRS_FILE, 'r') as f:
            data = json.load(f)
            return data.get('chats', {})
    except FileNotFoundError:
        return {}

def save_pairs(pairs):
    """Save pairs data with chat-specific tracking"""
    with open(PAIRS_FILE, 'w') as f:
        json.dump({'chats': pairs}, f, indent=4)

async def is_admin(update: Update) -> bool:
    """Check if the user is an admin of the current chat"""
    user_id = update.effective_user.id
    chat_member = await update.get_bot().get_chat_member(update.effective_chat.id, user_id)
    return chat_member.status in ['creator', 'administrator']

def initialize_filters():
    """Initialize event filters for all pairs across all chats"""
    global pair_filters
    pairs = load_pairs()
    
    # Create filters for all unique pairs across all chats
    unique_pairs = set()
    for chat_pairs in pairs.values():
        unique_pairs.update(chat_pairs.keys())
    
    for pair_address in unique_pairs:
        contract = w3.eth.contract(address=pair_address, abi=PAIR_ABI)
        pair_filters[pair_address] = contract.events.Swap.create_filter(from_block='latest')

def load_token_list():
    """Load the token list from file"""
    try:
        with open(TOKENS_FILE, 'r') as f:
            data = json.load(f)
            return data.get('tokens', {})
    except FileNotFoundError:
        print("Token list file not found")
        return {}

def load_base_assets():
    """Load base assets with their minimum quantities"""
    try:
        with open('baseassetlist.json', 'r') as f:
            data = json.load(f)
            return {asset['symbol']: asset['min_quantity'] for asset in data.get('baseassets', [])}
    except FileNotFoundError:
        print("Base asset list file not found")
        return {}

def get_token_priority(symbol, base_assets):
    """Get priority of token (lower number = higher priority)"""
    try:
        # Now base_assets is a dict, so we just check if the symbol is a key
        return 0 if symbol in base_assets else len(base_assets)
    except ValueError:
        return len(base_assets)  # Non-base assets have lowest priority

def is_buy_event(event, pair_data, token_list):
    """
    Determine if a swap is a buy based on base asset priorities.
    A swap is a buy if tokens are flowing from higher priority to lower priority token.
    """
    base_assets = load_base_assets()
    
    # Get token addresses from pair data
    token0_address = pair_data['token0']
    token1_address = pair_data['token1']
    
    # Get token symbols
    token0_symbol = token_list.get(token0_address, {}).get('symbol', 'UNKNOWN')
    token1_symbol = token_list.get(token1_address, {}).get('symbol', 'UNKNOWN')
    
    # Get token priorities
    token0_priority = get_token_priority(token0_symbol, base_assets)
    token1_priority = get_token_priority(token1_symbol, base_assets)
    
    # Get amounts from event
    amount0In = event['args']['amount0In']
    amount1In = event['args']['amount1In']
    amount0Out = event['args']['amount0Out']
    amount1Out = event['args']['amount1Out']
    
    # If token0 has higher priority (lower number)
    if token0_priority < token1_priority:
        # It's a buy if we're spending token0 (amount0In > 0)
        return amount0In > 0
    # If token1 has higher priority
    elif token1_priority < token0_priority:
        # It's a buy if we're spending token1 (amount1In > 0)
        return amount1In > 0
    # If same priority, default to original logic
    else:
        return amount0Out > 0

def format_message(event, pair_data):
    sender = event['args']['sender']
    to = event['args']['to']
    amount0In = event['args']['amount0In']
    amount1In = event['args']['amount1In']
    amount0Out = event['args']['amount0Out']
    amount1Out = event['args']['amount1Out']
    
    # Get transaction hash from event
    tx_hash = event['transactionHash'].hex()
    
    # Get pair name and image from pair data
    pair_name = pair_data['name'] if isinstance(pair_data, dict) else pair_data
    image_id = pair_data.get('image_id') if isinstance(pair_data, dict) else None
    
    # Get token addresses
    token0_address = pair_data['token0']
    token1_address = pair_data['token1']
    
    # Load token information
    token_list = load_token_list()
    token0_info = token_list.get(token0_address, {})
    token1_info = token_list.get(token1_address, {})
    
    # Get token symbols
    token0_symbol = token0_info.get('symbol', 'UNKNOWN')
    token1_symbol = token1_info.get('symbol', 'UNKNOWN')
    
    # Format amounts with decimals
    decimals0 = int(token0_info.get('decimals', 18))
    decimals1 = int(token1_info.get('decimals', 18))
    
    formatted_amount0 = float(amount0In or amount0Out) / (10 ** decimals0)
    formatted_amount1 = float(amount1In or amount1Out) / (10 ** decimals1)
    
    # Determine which token is being bought/sold and get its address
    if amount0In > 0:
        spent_amount = formatted_amount0
        spent_token = token0_symbol
        bought_amount = formatted_amount1
        bought_token = token1_symbol
        bought_token_address = token1_address
    else:
        spent_amount = formatted_amount1
        spent_token = token1_symbol
        bought_amount = formatted_amount0
        bought_token = token0_symbol
        bought_token_address = token0_address
    
    # Create a shortened version of addresses
    short_sender = f"{sender[:6]}...{sender[-4:]}"
    short_receiver = f"{to[:6]}...{to[-4:]}"
    
    # Calculate the equivalent amount in USD if available
    usd_value = ""
    if token0_info.get('price') is not None and amount0In > 0:
        usd_amount = formatted_amount0 * float(token0_info['price'])
        usd_value = f"\n💵 Value: ${usd_amount:,.2f} USD"
    elif token1_info.get('price') is not None and amount1In > 0:
        usd_amount = formatted_amount1 * float(token1_info['price'])
        usd_value = f"\n💵 Value: ${usd_amount:,.2f} USD"
    
    # Escape special characters for MarkdownV2
    def escape_md(text):
        # Characters that need escaping in MarkdownV2
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = str(text).replace(char, f'\\{char}')
        return text
    
    # Format numbers with escaped dots
    def format_number(num):
        return escape_md(f"{num:,.4f}")
    
    # Escape all variables that might contain special characters
    pair_name = escape_md(pair_name)
    spent_token = escape_md(spent_token)
    bought_token = escape_md(bought_token)
    short_sender = escape_md(short_sender)
    short_receiver = escape_md(short_receiver)
    tx_hash = escape_md(tx_hash)
    bought_token_address = escape_md(bought_token_address)
    
    # Format USD value with escaped characters
    if usd_value:
        usd_amount_formatted = escape_md(f"${usd_amount:,.2f}")
        usd_value = f"\n💵 Value: *{usd_amount_formatted} USD*"
    
    # Format the message with MarkdownV2 styling and emojis
    message = (
        f"*🚨 NEW TRADE ALERT\\! 🚨*\n\n"
        f"📊 Pair: *{pair_name}*\n"
        f"🔄 Trade Type: *BUY*\n\n"
        f"💰 *Trade Details:*\n"
        f"➡️ Spent: *{format_number(spent_amount)} {spent_token}*\n"
        f"⬅️ Bought: *{format_number(bought_amount)} {bought_token}*{usd_value}\n\n"
        f"👤 Sender: `{short_sender}`\n"
        f"🎯 Receiver: `{short_receiver}`\n\n"
        f"🔍 [View Transaction](https://bscscan\\.com/tx/0x{tx_hash})\n\n"
        f"🌊 [Buy {bought_token} on TidalDex](https://tidaldex\\.com/swap?outputCurrency={bought_token_address})"
    )
    
    return message, image_id

async def monitor_pairs(context: ContextTypes.DEFAULT_TYPE):
    """Monitor all tracked pairs and send updates to respective chats"""
    bot = context.bot
    pairs = load_pairs()
    token_list = load_token_list()
    base_assets = load_base_assets()
    
    # Check for new events using existing filters
    for pair_address, event_filter in pair_filters.items():
        # Find which chats are tracking this pair
        tracking_chats = [
            chat_id for chat_id, chat_pairs in pairs.items()
            if pair_address in chat_pairs
        ]
        
        if not tracking_chats:
            continue  # Skip if no chats are tracking this pair
            
        for event in event_filter.get_new_entries():
            # Find the pair data from any chat (they should all be the same)
            pair_data = next(
                (chat_pairs[pair_address] for chat_pairs in pairs.values() if pair_address in chat_pairs),
                None
            )
            
            if not pair_data:
                continue
                
            if is_buy_event(event, pair_data, token_list):
                # Check if the trade meets minimum quantity requirements
                if should_show_trade(event, pair_data, token_list, base_assets):
                    message, image_id = format_message(event, pair_data)
                    
                    # Send to all chats that are tracking this pair
                    for chat_id in tracking_chats:
                        try:
                            if image_id:
                                await bot.send_photo(
                                    chat_id=int(chat_id),
                                    photo=image_id,
                                    caption=message,
                                    parse_mode='MarkdownV2'
                                )
                            else:
                                await bot.send_message(
                                    chat_id=int(chat_id),
                                    text=message,
                                    parse_mode='MarkdownV2'
                                )
                        except Exception as e:
                            print(f"Error sending message to chat {chat_id}: {e}")

async def update_token_list(context: ContextTypes.DEFAULT_TYPE):
    """Update the token list if a new version is available"""
    print('Updating token list')
    try:
        download_token_list()
    except Exception as e:
        print(f"Error updating token list: {e}")

async def start_set_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start the image setting process"""
    if not await is_admin(update):
        await update.message.reply_text("You don't have permission to use this command.")
        return ConversationHandler.END
    
    # Check if pair address is provided
    if not context.args:
        await update.message.reply_text("Please provide a pair address: /setimage <pair_address>")
        return ConversationHandler.END
    
    pair_address = context.args[0]
    pairs = load_pairs()
    
    if pair_address not in pairs:
        await update.message.reply_text("This pair is not being tracked.")
        return ConversationHandler.END
    
    context.user_data['current_pair'] = pair_address
    await update.message.reply_text("Please send the image you want to use for this pair.")
    return WAITING_FOR_IMAGE

async def save_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Save the image ID for the pair"""
    photo = update.message.photo[-1]  # Get the largest size
    pair_address = context.user_data.get('current_pair')
    
    if not pair_address:
        await update.message.reply_text("Something went wrong. Please try again.")
        return ConversationHandler.END
    
    pairs = load_pairs()
    if isinstance(pairs[pair_address], str):
        pairs[pair_address] = {
            'name': pairs[pair_address],
            'image_id': photo.file_id
        }
    else:
        pairs[pair_address]['image_id'] = photo.file_id
    
    save_pairs(pairs)
    await update.message.reply_text("Image has been saved for this pair!")
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel the current operation"""
    await update.message.reply_text("Operation cancelled.")
    return ConversationHandler.END

async def generate_pair_image(token0_symbol: str, token1_symbol: str) -> str:
    """Generate an image for the trading pair using OpenAI's DALL-E and overlay token logos"""
    try:
        # Load token information to get logos
        token_list = load_token_list()
        base_assets = load_base_assets()
        
        # Get token priorities
        token0_priority = get_token_priority(token0_symbol, base_assets)
        token1_priority = get_token_priority(token1_symbol, base_assets)
        
        # Determine which token to show based on priority
        show_token0 = token0_priority > token1_priority
        token_to_show = token0_symbol if show_token0 else token1_symbol
        
        # Find token info for the token to show
        token_info = None
        for token_address, info in token_list.items():
            if info.get('symbol') == token_to_show:
                token_info = info
                break
        
        # Random style elements
        color_schemes = [
            "bioluminescent blue and coral pink",
            "deep sea purple and tropical orange",
            "aquamarine and phosphorescent green",
            "ocean sunset gold and teal",
            "pearl iridescent and deep navy",
            "coral reef rainbow spectrum",
            "arctic blue and tropical lagoon",
            "abyssal black and bioluminescent neon",
            "tide pool prismatic shimmer",
            "underwater aurora colors"
        ]

        backgrounds = [
            "a crystalline coral reef city",
            "an ancient underwater temple",
            "a bioluminescent deep sea trench",
            "a majestic underwater volcano",
            "swirling ocean currents and eddies",
            "a kelp forest stretching to infinity",
            "massive tidal waves frozen in time",
            "an underwater crystal cavern",
            "a sunken technological metropolis",
            "a mystical underwater vortex"
        ]

        creatures = [
            "a majestic humpback whale",
            "an ethereal manta ray",
            "a cybernetic giant crab",
            "a school of glowing jellyfish",
            "a colossal octopus",
            "a pod of bioluminescent dolphins",
            "a mythical sea serpent",
            "an ancient sea turtle",
            "a swarm of chromatic seahorses",
            "a legendary kraken"
        ]

        effects = [
            "surrounded by swirling tide pools",
            "emanating bioluminescent energy",
            "with trailing bubbles and sea foam",
            "radiating prismatic water patterns",
            "creating powerful tidal waves",
            "phasing through underwater currents",
            "casting reflections in the coral",
            "swimming through liquid light",
            "with flowing seaweed and algae",
            "wrapped in shimmering water ribbons"
        ]

        # Randomly select elements
        color = random.choice(color_schemes)
        background = random.choice(backgrounds)
        creature = random.choice(creatures)
        effect = random.choice(effects)
        
        # Create a dynamic prompt incorporating token names and random elements
        prompt = (
            f"Create a wide format underwater trading pair visualization for {token0_symbol} and {token1_symbol}. "
            f"The scene features {creature} on the left and right sides, emerging from {background}, {effect}. "
            f"The entire scene is bathed in {color} colors, creating an enchanting underwater atmosphere. "
            f"with a large, expansive clear space in the center"
            f"The style should be ultra-modern, high-contrast, and mesmerizing, "
            f"perfect for a cryptocurrency trading interface."
            f"Empty circular area in the center."
        )

        # Generate base image using DALL-E
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1792x1024",
            quality="standard",
            n=1,
        )

        # Get the image URL and download the base image
        image_url = response.data[0].url
        image_response = requests.get(image_url,timeout=270)
        if image_response.status_code != 200:
            raise Exception("Failed to download generated image")

        # Open the base image with PIL
        base_image = Image.open(BytesIO(image_response.content))
        
        # Calculate logo size based on image dimensions (45% of image height instead of 50%)
        logo_size = int(base_image.height * 0.45)  # This will be about 461px for 1024px height
        
        # Function to download and prepare logo
        def get_logo_image(logo_uri):
            if not logo_uri:
                return None
            try:
                response = requests.get(logo_uri,timeout=270)
                if response.status_code == 200:
                    logo = Image.open(BytesIO(response.content))
                    # Convert to RGBA if not already
                    if logo.mode != 'RGBA':
                        logo = logo.convert('RGBA')
                    # Resize to calculated size while maintaining aspect ratio
                    logo = logo.resize((logo_size, logo_size), Image.Resampling.LANCZOS)
                    return logo
            except Exception as e:
                print(f"Error downloading logo: {e}")
            return None

        # Download and prepare logo for the token to show
        logo = get_logo_image(token_info.get('logoURI')) if token_info else None

        if logo:
            # Convert PIL images to cv2 format
            base_cv = cv2.cvtColor(np.array(base_image), cv2.COLOR_RGB2BGR)
            logo_cv = cv2.cvtColor(np.array(logo), cv2.COLOR_RGBA2BGRA)

            # Calculate center position
            center_x = int(base_cv.shape[1] * 0.5) - logo_size//2
            center_y = int(base_cv.shape[0] * 0.5) - logo_size//2

            # Create a circular mask for the blur region
            circle_mask = np.zeros((logo_size, logo_size), dtype=np.uint8)
            cv2.circle(circle_mask, 
                      (logo_size//2, logo_size//2), 
                      int(logo_size * 0.55),  # Slightly larger radius for smoother transition
                      255, 
                      -1)  # Filled circle
            circle_mask = cv2.GaussianBlur(circle_mask, (31, 31), 0)  # Increased blur for softer edges

            # Extract the region to blur
            roi = base_cv[center_y:center_y+logo_size, center_x:center_x+logo_size].astype(np.float32)
            
            # Create blurred version of the ROI with stronger blur
            blurred_roi = cv2.GaussianBlur(roi, (35, 35), 0)
            
            # Brighten the blurred ROI slightly
            blurred_roi = cv2.multiply(blurred_roi, 1.2)  # Increase brightness by 20%
            blurred_roi = np.clip(blurred_roi, 0, 255)
            
            # Convert circle mask to float32 and expand to 3 channels
            circle_mask = circle_mask.astype(np.float32) / 255.0
            circle_mask_3channel = np.stack([circle_mask, circle_mask, circle_mask], axis=2)
            
            # Blend original and blurred ROI
            roi = cv2.multiply(roi, (1 - circle_mask_3channel))
            blurred_roi = cv2.multiply(blurred_roi, circle_mask_3channel)
            roi = cv2.add(roi, blurred_roi)
            
            # Create a mask from the alpha channel and convert to float32
            alpha = logo_cv[:, :, 3].astype(np.float32) / 255.0
            alpha = cv2.GaussianBlur(alpha, (7, 7), 0)  # Softer edge blur
            
            # Create a 3-channel alpha mask
            alpha_3channel = np.stack([alpha, alpha, alpha], axis=2)

            # Extract the logo's color channels (without alpha) and convert to float32
            logo_bgr = logo_cv[:, :, :3].astype(np.float32)

            # Adjust logo brightness (reduced from 1.4 to 1.1)
            logo_bgr = cv2.multiply(logo_bgr, 1.1)  # Increase brightness by 10%
            logo_bgr = np.clip(logo_bgr, 0, 255)

            # Add a subtle white glow behind the logo
            glow_mask = cv2.GaussianBlur(alpha, (21, 21), 0)
            glow_mask_3channel = np.stack([glow_mask, glow_mask, glow_mask], axis=2)
            white_glow = np.full_like(logo_bgr, 255.0)  # Create white glow
            roi = cv2.multiply(roi, (1 - glow_mask_3channel * 0.2))  # Reduce background where glow will be (reduced from 0.3)
            roi = cv2.add(roi, cv2.multiply(white_glow, glow_mask_3channel * 0.2))  # Add white glow (reduced from 0.3)

            # Blend the logo with adjusted opacity
            roi_blend = cv2.multiply(roi, (1 - alpha_3channel * 0.95))  # Increased background visibility
            logo_blend = cv2.multiply(logo_bgr, alpha_3channel * 0.95)  # Slightly reduced logo opacity
            final_roi = cv2.add(roi_blend, logo_blend)

            # Convert back to uint8 before placing back in the image
            final_roi = final_roi.astype(np.uint8)
            base_cv[center_y:center_y+logo_size, center_x:center_x+logo_size] = final_roi

            # Convert back to PIL Image
            base_image = Image.fromarray(cv2.cvtColor(base_cv, cv2.COLOR_BGR2RGB))

        # Convert the final image to bytes
        output = BytesIO()
        base_image.save(output, format='PNG')
        output.seek(0)
        return output.getvalue()

    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None

async def add_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Add a new pair to track for the current chat"""
    if not await is_admin(update):
        await update.message.reply_text("You don't have permission to use this command.")
        return
    
    if not context.args:
        await update.message.reply_text("Please provide pair address: /addpair <address>")
        return
    
    try:
        chat_id = str(update.effective_chat.id)
        # Convert address to checksum format
        address = Web3.to_checksum_address(context.args[0])
        
        # Create contract instance
        contract = w3.eth.contract(address=address, abi=PAIR_ABI)
        
        # Get token addresses from the pair contract
        token0_address = contract.functions.token0().call()
        token1_address = contract.functions.token1().call()
        
        # Load token information
        token_list = load_token_list()
        base_assets = load_base_assets()
        
        # Get token symbols
        token0_symbol = token_list.get(token0_address, {}).get('symbol', 'UNKNOWN')
        token1_symbol = token_list.get(token1_address, {}).get('symbol', 'UNKNOWN')
        
        # Get token priorities
        token0_priority = get_token_priority(token0_symbol, base_assets)
        token1_priority = get_token_priority(token1_symbol, base_assets)
        
        # Order tokens by priority (lower number = higher priority)
        if token0_priority <= token1_priority:
            first_symbol = token0_symbol
            second_symbol = token1_symbol
        else:
            first_symbol = token1_symbol
            second_symbol = token0_symbol
        
        # Generate pair name with ordered symbols
        pair_name = f"{first_symbol}-{second_symbol}"
        
        # Load all pairs
        pairs = load_pairs()
        
        # Initialize chat entry if it doesn't exist
        if chat_id not in pairs:
            pairs[chat_id] = {}
        
        # Add the pair to the chat's tracked pairs
        pairs[chat_id][address] = {
            'name': pair_name,
            'token0': token0_address,
            'token1': token1_address
        }
        save_pairs(pairs)
        
        # Initialize filter for new pair if not already exists
        if address not in pair_filters:
            pair_filters[address] = contract.events.Swap.create_filter(from_block='latest')
        
        # Notify that pair was added
        await update.message.reply_text(f"Added pair {pair_name}! Generating a unique image in the background...")
        
        # Generate image asynchronously
        async def generate_and_save_image():
            try:
                # Use the ordered symbols for image generation
                image_data = await generate_pair_image(first_symbol, second_symbol)
                if image_data:
                    # Send the image to Telegram and get the file_id
                    photo_message = await context.bot.send_photo(
                        chat_id=update.effective_chat.id,
                        photo=BytesIO(image_data),
                        caption=f"Generated image for {pair_name}"
                    )
                    image_id = photo_message.photo[-1].file_id
                    
                    # Update pair with the image
                    pairs = load_pairs()
                    if chat_id in pairs and address in pairs[chat_id]:  # Check if pair still exists
                        pairs[chat_id][address]['image_id'] = image_id
                        save_pairs(pairs)
                else:
                    await update.message.reply_text(
                        f"Note: Failed to generate custom image for {pair_name}. "
                        "You can add one later using /setimage."
                    )
            except Exception as e:
                await update.message.reply_text(
                    f"Note: Failed to generate custom image for {pair_name}: {str(e)}. "
                    "You can add one later using /setimage."
                )
        
        # Start image generation in the background
        asyncio.create_task(generate_and_save_image())
        
    except Exception as e:
        await update.message.reply_text(f"Error adding pair: {str(e)}")
        return

async def remove_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Remove a pair from tracking for the current chat"""
    if not await is_admin(update):
        await update.message.reply_text("You don't have permission to use this command.")
        return
    
    if not context.args:
        await update.message.reply_text("Please provide pair address: /removepair <address>")
        return
    
    chat_id = str(update.effective_chat.id)
    address = context.args[0]
    pairs = load_pairs()
    
    if chat_id in pairs and address in pairs[chat_id]:
        del pairs[chat_id][address]
        # Remove empty chat entry
        if not pairs[chat_id]:
            del pairs[chat_id]
        save_pairs(pairs)
        
        # Check if pair is used by other chats before removing filter
        pair_in_use = any(address in chat_pairs for chat_id, chat_pairs in pairs.items())
        if not pair_in_use and address in pair_filters:
            del pair_filters[address]
            
        await update.message.reply_text(f"Removed pair {address}")
    else:
        await update.message.reply_text("Pair not found in this chat")

async def list_pairs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List all tracked pairs for the current chat"""
    chat_id = str(update.effective_chat.id)
    pairs = load_pairs()
    
    if chat_id not in pairs or not pairs[chat_id]:
        await update.message.reply_text("No pairs are being tracked in this chat.")
        return
    
    message = "Tracked pairs in this chat:\n\n"
    for address, data in pairs[chat_id].items():
        name = data['name'] if isinstance(data, dict) else data
        message += f"{name}: {address}\n"
    
    await update.message.reply_text(message)

def should_show_trade(event, pair_data, token_list, base_assets):
    """
    Check if a trade should be shown based on base asset minimum quantities.
    Returns True if the trade should be shown, False if it should be filtered out.
    """
    # Get token addresses from pair data
    token0_address = pair_data['token0']
    token1_address = pair_data['token1']
    
    # Get token symbols and decimals
    token0_info = token_list.get(token0_address, {})
    token1_info = token_list.get(token1_address, {})
    token0_symbol = token0_info.get('symbol', 'UNKNOWN')
    token1_symbol = token1_info.get('symbol', 'UNKNOWN')
    decimals0 = int(token0_info.get('decimals', 18))
    decimals1 = int(token1_info.get('decimals', 18))
    
    # Get amounts from event
    amount0In = float(event['args']['amount0In']) / (10 ** decimals0)
    amount1In = float(event['args']['amount1In']) / (10 ** decimals1)
    amount0Out = float(event['args']['amount0Out']) / (10 ** decimals0)
    amount1Out = float(event['args']['amount1Out']) / (10 ** decimals1)
    
    # Check if either token is a base asset
    if token0_symbol in base_assets:
        min_quantity = base_assets[token0_symbol]
        if amount0In > 0 and amount0In < min_quantity:
            return False
        if amount0Out > 0 and amount0Out < min_quantity:
            return False
    
    if token1_symbol in base_assets:
        min_quantity = base_assets[token1_symbol]
        if amount1In > 0 and amount1In < min_quantity:
            return False
        if amount1Out > 0 and amount1Out < min_quantity:
            return False
    
    return True

async def regenerate_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Regenerate the image for a specific pair"""
    if not await is_admin(update):
        await update.message.reply_text("You don't have permission to use this command.")
        return
    
    if not context.args:
        await update.message.reply_text("Please provide pair address: /regenimg <address>")
        return
    
    try:
        chat_id = str(update.effective_chat.id)
        # Convert address to checksum format
        address = Web3.to_checksum_address(context.args[0])
        pairs = load_pairs()
        
        if chat_id not in pairs or address not in pairs[chat_id]:
            await update.message.reply_text("This pair is not being tracked in this chat.")
            return
        
        pair_data = pairs[chat_id][address]
        
        # Load token information
        token_list = load_token_list()
        base_assets = load_base_assets()
        
        # Get token symbols
        token0_symbol = token_list.get(pair_data['token0'], {}).get('symbol', 'UNKNOWN')
        token1_symbol = token_list.get(pair_data['token1'], {}).get('symbol', 'UNKNOWN')
        
        # Get token priorities
        token0_priority = get_token_priority(token0_symbol, base_assets)
        token1_priority = get_token_priority(token1_symbol, base_assets)
        
        # Order tokens by priority (lower number = higher priority)
        if token0_priority <= token1_priority:
            first_symbol = token0_symbol
            second_symbol = token1_symbol
        else:
            first_symbol = token1_symbol
            second_symbol = token0_symbol
        
        await update.message.reply_text(f"Regenerating image for {pair_data['name']}...")
        
        try:
            # Generate new image
            image_data = await generate_pair_image(first_symbol, second_symbol)
            if image_data:
                # Send the image to Telegram and get the file_id
                photo_message = await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=BytesIO(image_data),
                    caption=f"Regenerated image for {pair_data['name']}"
                )
                
                # Get the image ID and save it immediately after successful sending
                image_id = photo_message.photo[-1].file_id
                
                # Reload pairs to get the latest data
                pairs = load_pairs()
                if chat_id in pairs and address in pairs[chat_id]:
                    pairs[chat_id][address]['image_id'] = image_id
                    save_pairs(pairs)
                    await update.message.reply_text("Image has been updated and saved successfully!")
                else:
                    await update.message.reply_text("Warning: Image was generated but could not be saved because the pair is no longer being tracked.")
            else:
                await update.message.reply_text("Failed to generate new image.")
                
        except asyncio.TimeoutError:
            # Handle timeout specifically
            await update.message.reply_text("Operation timed out. Checking if the image was saved...")
            
            # Check if the image was actually sent and saved
            pairs = load_pairs()
            if chat_id in pairs and address in pairs[chat_id] and 'image_id' in pairs[chat_id][address]:
                await update.message.reply_text("Image was successfully saved despite the timeout.")
            else:
                await update.message.reply_text("Image generation was successful but saving failed. Please try the command again.")
            
    except Exception as e:
        await update.message.reply_text(f"Error regenerating image: {str(e)}")

async def redownload_tokens(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Force redownload of the token list"""
    if not await is_admin(update):
        await update.message.reply_text("You don't have permission to use this command.")
        return
    
    try:
        await update.message.reply_text("Downloading token list...")
        download_token_list()
        await update.message.reply_text("Token list has been successfully updated!")
    except Exception as e:
        await update.message.reply_text(f"Error updating token list: {str(e)}")

def main():
    """Set up the bot and start monitoring pairs"""
    # Build application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Initialize filters for existing pairs
    initialize_filters()
    
    # Create conversation handler for image setting
    image_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("setimage", start_set_image)],
        states={
            WAITING_FOR_IMAGE: [
                MessageHandler(filters.PHOTO, save_image),
                CommandHandler("cancel", cancel)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    
    # Add command handlers
    application.add_handler(CommandHandler("addpair", add_pair))
    application.add_handler(CommandHandler("removepair", remove_pair))
    application.add_handler(CommandHandler("listpairs", list_pairs))
    application.add_handler(CommandHandler("regenimg", regenerate_image))
    application.add_handler(CommandHandler("reloadtokens", redownload_tokens))
    application.add_handler(image_conv_handler)
    
    # Add the monitoring job
    job_queue = application.job_queue
    job_queue.run_repeating(monitor_pairs, interval=5, first=5)
    
    # Add token list update job (every hour)
    job_queue.run_repeating(update_token_list, interval=3600, first=5)
    
    # Run the application
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Bot stopped gracefully")
