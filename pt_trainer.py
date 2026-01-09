# Unbuffered output for real-time GUI updates
import sys
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

import time
"""
<------------
newest oldest
------------>
oldest newest
"""
avg50 = []
import datetime
import traceback
import linecache
import base64
import calendar
import hashlib
import hmac
from datetime import datetime
sells_count = 0
prediction_prices_avg_list = []
pt_server = 'server'
import psutil
import logging

# NumPy & Numba imports with graceful fallback
try:
	import numpy as np
	NUMPY_AVAILABLE = True
except ImportError:
	NUMPY_AVAILABLE = False
	print("Warning: NumPy not available. Performance will be reduced.")

try:
	from numba import jit
	NUMBA_AVAILABLE = True
except ImportError:
	NUMBA_AVAILABLE = False
	print("Warning: Numba not available. Performance will be reduced.")
	# Create a no-op decorator
	def jit(*args, **kwargs):
		def decorator(func):
			return func
		return decorator

# Binance API imports
import requests
list_len = 0
restarting = 'no'
in_trade = 'no'
updowncount = 0
updowncount1 = 0
updowncount1_2 = 0
updowncount1_3 = 0
updowncount1_4 = 0
high_var2 = 0.0
low_var2 = 0.0
last_flipped = 'no'
starting_amounth02 = 100.0
starting_amounth05 = 100.0
starting_amounth10 = 100.0
starting_amounth20 = 100.0
starting_amounth50 = 100.0
starting_amount = 100.0
starting_amount1 = 100.0
starting_amount1_2 = 100.0
starting_amount1_3 = 100.0
starting_amount1_4 = 100.0
starting_amount2 = 100.0
starting_amount2_2 = 100.0
starting_amount2_3 = 100.0
starting_amount2_4 = 100.0
starting_amount3 = 100.0
starting_amount3_2 = 100.0
starting_amount3_3 = 100.0
starting_amount3_4 = 100.0
starting_amount4 = 100.0
starting_amount4_2 = 100.0
starting_amount4_3 = 100.0
starting_amount4_4 = 100.0
profit_list = []
profit_list1 = []
profit_list1_2 = []
profit_list1_3 = []
profit_list1_4 = []
profit_list2 = []
profit_list2_2 = []
profit_list2_3 = []
profit_list2_4 = []
profit_list3 = []
profit_list3_2 = []
profit_list3_3 = []
profit_list4 = []
profit_list4_2 = []
good_hits = []
good_preds = []
good_preds2 = []
good_preds3 = []
good_preds4 = []
good_preds5 = []
good_preds6 = []
big_good_preds = []
big_good_preds2 = []
big_good_preds3 = []
big_good_preds4 = []
big_good_preds5 = []
big_good_preds6 = []
big_good_hits = []
upordown = []
upordown1 = []
upordown1_2 = []
upordown1_3 = []
upordown1_4 = []
upordown2 = []
upordown2_2 = []
upordown2_3 = []
upordown2_4 = []
upordown3 = []
upordown3_2 = []
upordown3_3 = []
upordown3_4 = []
upordown4 = []
upordown4_2 = []
upordown4_3 = []
upordown4_4 = []
upordown5 = []
import json
import uuid
import os

# ---- speed knobs ----
VERBOSE = False  # set True if you want the old high-volume prints
def vprint(*args, **kwargs):
	if VERBOSE:
		print(*args, **kwargs)

# Cache memory/weights in RAM (avoid re-reading and re-writing every loop)
_memory_cache = {}  # tf_choice -> dict(memory_list, weight_list, high_weight_list, low_weight_list, dirty)
_last_threshold_written = {}  # tf_choice -> float

def _read_text(path):
	with open(path, "r", encoding="utf-8", errors="ignore") as f:
		return f.read()

def load_memory(tf_choice):
	"""Load memories/weights for a timeframe once and keep them in RAM."""
	if tf_choice in _memory_cache:
		return _memory_cache[tf_choice]
	data = {
		"memory_list": [],
		"weight_list": [],
		"high_weight_list": [],
		"low_weight_list": [],
		"dirty": False,
	}
	try:
		data["memory_list"] = _read_text(f"memories_{tf_choice}.txt").replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split('~')
	except:
		data["memory_list"] = []
	try:
		data["weight_list"] = _read_text(f"memory_weights_{tf_choice}.txt").replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split(' ')
	except:
		data["weight_list"] = []
	try:
		data["high_weight_list"] = _read_text(f"memory_weights_high_{tf_choice}.txt").replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split(' ')
	except:
		data["high_weight_list"] = []
	try:
		data["low_weight_list"] = _read_text(f"memory_weights_low_{tf_choice}.txt").replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split(' ')
	except:
		data["low_weight_list"] = []
	_memory_cache[tf_choice] = data
	return data

def flush_memory(tf_choice, force=False):
	"""Write memories/weights back to disk only when they changed (batch IO)."""
	data = _memory_cache.get(tf_choice)
	if not data:
		return
	if (not data.get("dirty")) and (not force):
		return
	try:
		with open(f"memories_{tf_choice}.txt", "w+", encoding="utf-8") as f:
			f.write("~".join([x for x in data["memory_list"] if str(x).strip() != ""]))
	except:
		pass
	try:
		with open(f"memory_weights_{tf_choice}.txt", "w+", encoding="utf-8") as f:
			f.write(" ".join([str(x) for x in data["weight_list"] if str(x).strip() != ""]))
	except:
		pass
	try:
		with open(f"memory_weights_high_{tf_choice}.txt", "w+", encoding="utf-8") as f:
			f.write(" ".join([str(x) for x in data["high_weight_list"] if str(x).strip() != ""]))
	except:
		pass
	try:
		with open(f"memory_weights_low_{tf_choice}.txt", "w+", encoding="utf-8") as f:
			f.write(" ".join([str(x) for x in data["low_weight_list"] if str(x).strip() != ""]))
	except:
		pass
	data["dirty"] = False

def write_threshold_sometimes(tf_choice, perfect_threshold, loop_i, every=200):
	"""Avoid writing neural_perfect_threshold_* every single loop."""
	last = _last_threshold_written.get(tf_choice)
	# write occasionally, or if it changed meaningfully
	if (loop_i % every != 0) and (last is not None) and (abs(perfect_threshold - last) < 0.05):
		return
	try:
		with open(f"neural_perfect_threshold_{tf_choice}.txt", "w+", encoding="utf-8") as f:
			f.write(str(perfect_threshold))
		_last_threshold_written[tf_choice] = perfect_threshold
	except:
		pass

def should_stop_training(loop_i, every=50):
	"""Check killer.txt less often (still responsive, way less IO)."""
	if loop_i % every != 0:
		return False
	try:
		with open("killer.txt", "r", encoding="utf-8", errors="ignore") as f:
			return f.read().strip().lower() == "yes"
	except:
		return False

def PrintException():
	exc_type, exc_obj, tb = sys.exc_info()
	f = tb.tb_frame
	lineno = tb.tb_lineno
	filename = f.f_code.co_filename
	linecache.checkcache(filename)
	line = linecache.getline(filename, lineno, f.f_globals)
	print ('EXCEPTION IN (LINE {} "{}"): {}'.format(lineno, line.strip(), exc_obj), flush=True)

# Enhanced exception logging
def log_exception_to_file(coin, timeframe, loop_i, window_info=None):
	"""Log exceptions to training_errors.jsonl for debugging."""
	try:
		error_entry = {
			"timestamp": time.time(),
			"coin": coin,
			"timeframe": timeframe,
			"loop_i": loop_i,
			"window_info": window_info,
			"exception": str(sys.exc_info()[1]),
			"traceback": traceback.format_exc()
		}
		with open("training_errors.jsonl", "a", encoding="utf-8") as f:
			f.write(json.dumps(error_entry) + "\n")
	except:
		pass

# Binance API functions
def get_binance_kline(symbol, interval, start_time=None, end_time=None, limit=1000):
	"""
	Fetch kline data from Binance API.
	Converts Binance format to Kucoin-compatible format.
	
	Args:
		symbol: Trading pair (e.g., 'BTCUSDT')
		interval: Timeframe ('1h', '2h', '4h', '8h', '12h', '1d', '1w')
		start_time: Start timestamp in seconds
		end_time: End timestamp in seconds
		limit: Maximum number of candles (max 1000)
	
	Returns:
		List of candles in Kucoin-compatible format
	"""
	# Timeframe mapping: Kucoin -> Binance
	tf_map = {
		'1hour': '1h',
		'2hour': '2h',
		'4hour': '4h',
		'8hour': '8h',
		'12hour': '12h',
		'1day': '1d',
		'1week': '1w'
	}
	
	binance_interval = tf_map.get(interval, interval)
	
	# Convert symbol: BTC-USDT -> BTCUSDT
	binance_symbol = symbol.replace('-', '')
	
	url = f"https://api.binance.com/api/v3/klines"
	params = {
		'symbol': binance_symbol,
		'interval': binance_interval,
		'limit': min(limit, 1000)  # Binance max is 1000
	}
	
	if start_time:
		params['startTime'] = int(start_time * 1000)  # Binance uses milliseconds
	if end_time:
		params['endTime'] = int(end_time * 1000)
	
	try:
		response = requests.get(url, params=params, timeout=10)
		response.raise_for_status()
		data = response.json()
		
		# Convert Binance format to Kucoin-compatible format
		# Binance: [timestamp, open, high, low, close, volume, ...]
		# Kucoin: [timestamp, open, close, high, low, volume]
		result = []
		for candle in data:
			timestamp = int(candle[0] / 1000)  # Convert milliseconds to seconds
			open_price = float(candle[1])
			high_price = float(candle[2])
			low_price = float(candle[3])
			close_price = float(candle[4])
			volume = float(candle[5])
			
			# Format as Kucoin-style string: [timestamp, open, close, high, low, volume]
			candle_str = f"[{timestamp}, {open_price}, {close_price}, {high_price}, {low_price}, {volume}]"
			result.append(candle_str)
		
		return result
	except Exception as e:
		print(f"Binance API error: {e}", flush=True)
		raise

class BinanceMarket:
	"""Compatible interface with old Kucoin market object."""
	def __init__(self):
		pass
	
	def get_kline(self, symbol, interval, startAt=None, endAt=None):
		"""Get kline data - compatible with old Kucoin interface."""
		return get_binance_kline(symbol, interval, start_time=startAt, end_time=endAt)
	
	def get_ticker(self, symbol):
		"""Get current ticker price."""
		binance_symbol = symbol.replace('-', '')
		url = f"https://api.binance.com/api/v3/ticker/price"
		params = {'symbol': binance_symbol}
		try:
			response = requests.get(url, params=params, timeout=10)
			response.raise_for_status()
			data = response.json()
			# Return in Kucoin-compatible format
			return f'{{"price": {data["price"]}}}'
		except Exception as e:
			print(f"Binance ticker error: {e}", flush=True)
			return '{"price": 0}'

# Initialize Binance market (replaces Kucoin)
market = BinanceMarket()

# NumPy helper functions
def parse_memory_to_array(memory_str):
	"""Parse memory string to NumPy array for fast pattern matching."""
	if not NUMPY_AVAILABLE:
		# Fallback to list
		return memory_str.replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split(' ')
	
	try:
		parts = memory_str.split('{}')[0].replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split(' ')
		return np.array([float(x) for x in parts if x.strip()], dtype=np.float64)
	except:
		return np.array([], dtype=np.float64)

def parse_memory_metadata(memory_str):
	"""Extract high_diff and low_diff from memory string."""
	try:
		parts = memory_str.split('{}')
		if len(parts) >= 3:
			high_diff = float(parts[1].replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').replace(' ',''))
			low_diff = float(parts[2].replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').replace(' ',''))
			return high_diff, low_diff
	except:
		pass
	return 0.0, 0.0

# Numba JIT functions
if NUMBA_AVAILABLE:
	@jit(nopython=True)
	def calculate_pattern_difference(current_pattern, memory_pattern):
		"""Calculate pattern difference using NumPy arrays (JIT compiled)."""
		if len(current_pattern) == 0 or len(memory_pattern) == 0:
			return 1000000.0
		
		checks = np.zeros(len(current_pattern))
		for i in range(len(current_pattern)):
			current_candle = current_pattern[i]
			memory_candle = memory_pattern[i]
			if current_candle + memory_candle == 0.0:
				checks[i] = 0.0
			else:
				checks[i] = abs((abs(current_candle - memory_candle) / ((current_candle + memory_candle) / 2)) * 100)
		
		return np.mean(checks)
	
	@jit(nopython=True)
	def vectorized_price_changes(price_list, open_price_list):
		"""Calculate price changes using vectorized operations (JIT compiled)."""
		if len(price_list) != len(open_price_list):
			# Return empty array with explicit dtype for Numba type inference
			# Use np.empty instead of np.array([]) for better Numba compatibility
			return np.empty(0, dtype=np.float64)
		
		# Convert lists to arrays - Numba can handle np.array() on lists in nopython mode
		price_arr = np.asarray(price_list, dtype=np.float64)
		open_arr = np.asarray(open_price_list, dtype=np.float64)
		
		# Calculate: 100 * ((price - open) / open)
		changes = 100 * ((price_arr - open_arr) / open_arr)
		return changes
else:
	# Fallback implementations without Numba
	def calculate_pattern_difference(current_pattern, memory_pattern):
		"""Fallback pattern difference calculation."""
		if len(current_pattern) == 0 or len(memory_pattern) == 0:
			return 1000000.0
		
		checks = []
		for i in range(len(current_pattern)):
			current_candle = float(current_pattern[i])
			memory_candle = float(memory_pattern[i])
			if current_candle + memory_candle == 0.0:
				checks.append(0.0)
			else:
				checks.append(abs((abs(current_candle - memory_candle) / ((current_candle + memory_candle) / 2)) * 100))
		
		return sum(checks) / len(checks) if checks else 1000000.0
	
	def vectorized_price_changes(price_list, open_price_list):
		"""Fallback price change calculation."""
		if len(price_list) != len(open_price_list):
			return []
		
		changes = []
		for i in range(len(price_list)):
			change = 100 * ((float(price_list[i]) - float(open_price_list[i])) / float(open_price_list[i]))
			changes.append(change)
		
		return changes

how_far_to_look_back = 100000
number_of_candles = [2]
number_of_candles_index = 0
restarted_yet = 0  # Initialize before use
def restart_program():
	"""Restarts the current program, with file objects and descriptors cleanup"""

	try:
		p = psutil.Process(os.getpid())
		for handler in p.open_files() + p.connections():
			os.close(handler.fd)
	except Exception as e:
		logging.error(e)
	python = sys.executable
	os.execl(python, python, * sys.argv)
try:
	if restarted_yet > 2:
		restarted_yet = 0
	else:	
		pass
except:
	restarted_yet = 0
tf_choices = ['1hour', '2hour', '4hour', '8hour', '12hour', '1day', '1week']
tf_minutes = [60, 120, 240, 480, 720, 1440, 10080]
# --- GUI HUB INPUT (NO PROMPTS) ---
# Usage: python pt_trainer.py BTC [reprocess_yes|reprocess_no] [--test|-t]
_arg_coin = "BTC"
_test_mode = False

try:
	if len(sys.argv) > 1 and str(sys.argv[1]).strip():
		_arg_coin = str(sys.argv[1]).strip().upper()
except Exception:
	_arg_coin = "BTC"

# Check for test mode flag
if '--test' in sys.argv or '-t' in sys.argv:
	_test_mode = True
	print("Test mode enabled - using limited data for faster testing", flush=True)

coin_choice = _arg_coin + '-USDT'

restart_processing = "yes"

# Price history cache system
def find_price_history_dir():
	"""Multi-path detection for price_history directory."""
	base_dir = os.getcwd()
	possible_paths = [
		os.path.join(base_dir, "price_history"),
		os.path.join(base_dir, "..", "price_history"),
		os.path.join(os.path.dirname(os.path.abspath(__file__)), "price_history"),
		os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "price_history"),
	]
	
	for path in possible_paths:
		if os.path.isdir(path):
			return path
	
	# Create price_history in current directory if not found
	try:
		os.makedirs("price_history", exist_ok=True)
		return "price_history"
	except:
		return None

def load_price_history_cache(coin, timeframe, last_start_time=0):
	"""Load price history from cache file."""
	cache_dir = find_price_history_dir()
	if not cache_dir:
		return []
	
	cache_file = os.path.join(cache_dir, coin, f"{timeframe}.json")
	
	if not os.path.isfile(cache_file):
		return []
	
	try:
		with open(cache_file, "r", encoding="utf-8") as f:
			cached_data = json.load(f)
		
		# Filter by time range
		filtered_data = []
		for candle in cached_data:
			if len(candle) >= 1:
				candle_time = int(candle[0])
				if candle_time >= last_start_time:
					filtered_data.append(candle)
		
		# Test mode: only use last 100 candles
		if _test_mode and len(filtered_data) > 100:
			filtered_data = filtered_data[-100:]
		
		# If filtering resulted in 0 results, use all cache data
		if len(filtered_data) == 0 and len(cached_data) > 0:
			if _test_mode and len(cached_data) > 100:
				filtered_data = cached_data[-100:]
			else:
				filtered_data = cached_data
		
		# Convert to Kucoin-compatible format
		result = []
		for candle in filtered_data:
			if len(candle) >= 6:
				candle_str = f"[{candle[0]}, {candle[1]}, {candle[2]}, {candle[3]}, {candle[4]}, {candle[5]}]"
				result.append(candle_str)
		
		print(f"Loaded {len(result)} candles from cache for {coin}/{timeframe}", flush=True)
		return result
	except Exception as e:
		print(f"Error loading cache: {e}", flush=True)
		return []

def save_price_history_cache(coin, timeframe, history_data):
	"""Save price history to cache file."""
	cache_dir = find_price_history_dir()
	if not cache_dir:
		return
	
	coin_dir = os.path.join(cache_dir, coin)
	try:
		os.makedirs(coin_dir, exist_ok=True)
	except:
		return
	
	cache_file = os.path.join(coin_dir, f"{timeframe}.json")
	
	# Convert Kucoin format to JSON array
	cached_data = []
	for candle_str in history_data:
		try:
			# Parse: [timestamp, open, close, high, low, volume]
			candle_str = candle_str.replace('[', '').replace(']', '').strip()
			parts = [p.strip() for p in candle_str.split(',')]
			if len(parts) >= 6:
				candle = [
					int(float(parts[0])),  # timestamp
					float(parts[1]),  # open
					float(parts[2]),  # close
					float(parts[3]),  # high
					float(parts[4]),  # low
					float(parts[5])   # volume
				]
				cached_data.append(candle)
		except:
			continue
	
	try:
		with open(cache_file, "w", encoding="utf-8") as f:
			json.dump(cached_data, f)
	except:
		pass

# GUI reads this status file to know if this coin is TRAINING or FINISHED
_trainer_started_at = int(time.time())
try:
	with open("trainer_status.json", "w", encoding="utf-8") as f:
		json.dump(
			{
				"coin": _arg_coin,
				"state": "TRAINING",
				"started_at": _trainer_started_at,
				"timestamp": _trainer_started_at,
			},
			f,
		)
except Exception:
	pass


the_big_index = 0
while True:
	list_len = 0
	restarting = 'no'
	in_trade = 'no'
	updowncount = 0
	updowncount1 = 0
	updowncount1_2 = 0
	updowncount1_3 = 0
	updowncount1_4 = 0
	high_var2 = 0.0
	low_var2 = 0.0
	last_flipped = 'no'
	starting_amounth02 = 100.0
	starting_amounth05 = 100.0
	starting_amounth10 = 100.0
	starting_amounth20 = 100.0
	starting_amounth50 = 100.0
	starting_amount = 100.0
	starting_amount1 = 100.0
	starting_amount1_2 = 100.0
	starting_amount1_3 = 100.0
	starting_amount1_4 = 100.0
	starting_amount2 = 100.0
	starting_amount2_2 = 100.0
	starting_amount2_3 = 100.0
	starting_amount2_4 = 100.0
	starting_amount3 = 100.0
	starting_amount3_2 = 100.0
	starting_amount3_3 = 100.0
	starting_amount3_4 = 100.0
	starting_amount4 = 100.0
	starting_amount4_2 = 100.0
	starting_amount4_3 = 100.0
	starting_amount4_4 = 100.0
	profit_list = []
	profit_list1 = []
	profit_list1_2 = []
	profit_list1_3 = []
	profit_list1_4 = []
	profit_list2 = []
	profit_list2_2 = []
	profit_list2_3 = []
	profit_list2_4 = []
	profit_list3 = []
	profit_list3_2 = []
	profit_list3_3 = []
	profit_list4 = []
	profit_list4_2 = []
	good_hits = []
	good_preds = []
	good_preds2 = []
	good_preds3 = []
	good_preds4 = []
	good_preds5 = []
	good_preds6 = []
	big_good_preds = []
	big_good_preds2 = []
	big_good_preds3 = []
	big_good_preds4 = []
	big_good_preds5 = []
	big_good_preds6 = []
	big_good_hits = []
	upordown = []
	upordown1 = []
	upordown1_2 = []
	upordown1_3 = []
	upordown1_4 = []
	upordown2 = []
	upordown2_2 = []
	upordown2_3 = []
	upordown2_4 = []
	upordown3 = []
	upordown3_2 = []
	upordown3_3 = []
	upordown3_4 = []
	upordown4 = []
	upordown4_2 = []
	upordown4_3 = []
	upordown4_4 = []
	upordown5 = []
	tf_choice = tf_choices[the_big_index]
	_mem = load_memory(tf_choice)
	memory_list = _mem["memory_list"]
	weight_list = _mem["weight_list"]
	high_weight_list = _mem["high_weight_list"]
	low_weight_list = _mem["low_weight_list"]
	no_list = 'no' if len(memory_list) > 0 else 'yes'

	tf_list = ['1hour',tf_choice,tf_choice]
	choice_index = tf_choices.index(tf_choice)
	minutes_list = [60,tf_minutes[choice_index],tf_minutes[choice_index]]
	if restarted_yet < 2:
		timeframe = tf_list[restarted_yet]#droplet setting (create list for all timeframes)
		timeframe_minutes = minutes_list[restarted_yet]#droplet setting (create list for all timeframe_minutes)
	else:
		timeframe = tf_list[2]#droplet setting (create list for all timeframes)
		timeframe_minutes = minutes_list[2]#droplet setting (create list for all timeframe_minutes)
	start_time = int(time.time())
	restarting = 'no'
	success_rate = 85
	volume_success_rate = 60
	candles_to_predict = 1#droplet setting (Max is half of number_of_candles)(Min is 2)
	max_difference = .5
	preferred_difference = .4 #droplet setting (max profit_margin) (Min 0.01)
	min_good_matches = 1#droplet setting (Max 100) (Min 4)
	max_good_matches = 1#droplet setting (Max 100) (Min is min_good_matches)
	prediction_expander = 1.33
	prediction_expander2 = 1.5
	prediction_adjuster = 0.0
	diff_avg_setting = 0.01
	min_success_rate = 90
	histories = 'off'
	coin_choice_index = 0
	list_of_ys_count = 0
	last_difference_between = 0.0
	history_list = []
	history_list2 = []
	len_avg = []
	list_len = 0
	start_time = int(time.time())
	start_time_yes = start_time
	if 'n' in restart_processing.lower():
		try:
			file = open('trainer_last_start_time.txt','r')
			last_start_time = int(file.read())
			file.close()
		except:
			last_start_time = 0.0
	else:
		last_start_time = 0.0
	end_time = int(start_time-((1500*timeframe_minutes)*60))
	
	# Try to load from cache first
	history_list = load_price_history_cache(_arg_coin, timeframe, last_start_time)
	
	# If cache is empty or insufficient, fetch from API
	if len(history_list) == 0:
		print('Fetching history from Binance API...', flush=True)
		perc_comp = format((len(history_list2)/how_far_to_look_back)*100,'.2f')
		last_perc_comp = perc_comp+'kjfjakjdakd'
		current_start_time = start_time
		current_end_time = end_time
		
		while True:
			time.sleep(.5)
			try:
				# Binance API max is 1000 candles per request
				history = market.get_kline(coin_choice, timeframe, startAt=current_end_time, endAt=current_start_time)
				if not history:
					break
			except Exception as e:
				PrintException()
				log_exception_to_file(_arg_coin, timeframe, 0, {"stage": "history_fetch", "start": current_start_time, "end": current_end_time})
				time.sleep(3.5)
				continue
			
			index = 0
			while True:
				if index >= len(history):
					break
				history_list.append(history[index])
				index += 1
			
			perc_comp = format((len(history_list)/how_far_to_look_back)*100,'.2f')
			print('gathering history', flush=True)
			current_change = len(history_list)-list_len	
			try:
				print('\n\n\n\n', flush=True)
				print(current_change, flush=True)
				# Binance returns max 1000 candles, so check if we got less
				if current_change < 1000:
					break
			except:
				PrintException()
				pass
			len_avg.append(current_change)
			list_len = len(history_list)
			last_perc_comp = perc_comp
			current_start_time = current_end_time
			current_end_time = int(current_start_time-((1500*timeframe_minutes)*60))
			print(last_start_time, flush=True)
			print(current_start_time, flush=True)
			print(current_end_time, flush=True)
			print('\n', flush=True)
			if current_start_time <= last_start_time:
				break
		
		# Save to cache
		if len(history_list) > 0:
			save_price_history_cache(_arg_coin, timeframe, history_list)
			print(f'Saved {len(history_list)} candles to cache', flush=True)
	else:
		print(f'Using {len(history_list)} candles from cache', flush=True)
	if timeframe == '1day' or timeframe == '1week':
		if restarted_yet == 0:
			index = int(len(history_list)/2)
		else:
			index = 1
	else:
		index = int(len(history_list)/2)
	price_list = []
	high_price_list = []
	low_price_list = []
	open_price_list = []
	volume_list = []
	minutes_passed = 0
	try:
		while True:
			working_minute = str(history_list[index]).replace('"','').replace("'","").split(", ")
			try:
				if index == 1:
					current_tf_time = float(working_minute[0].replace('[',''))
					last_tf_time = current_tf_time
				else:
					pass
				candle_time = float(working_minute[0].replace('[',''))
				openPrice = float(working_minute[1])                
				closePrice = float(working_minute[2])
				highPrice = float(working_minute[3])
				lowPrice = float(working_minute[4])
				open_price_list.append(openPrice)
				price_list.append(closePrice)
				high_price_list.append(highPrice)
				low_price_list.append(lowPrice)
				index += 1
				if index >= len(history_list):
					break
				else:
					continue
			except:
				PrintException()
				index += 1
				if index >= len(history_list):
					break
				else:
					continue
		open_price_list.reverse()
		price_list.reverse()
		high_price_list.reverse()
		low_price_list.reverse()
		ticker_data = str(market.get_ticker(coin_choice)).replace('"','').replace("'","").replace("[","").replace("{","").replace("]","").replace("}","").replace(",","").lower().split(' ')
		price = float(ticker_data[ticker_data.index('price:')+1])
	except:
		PrintException()
	history_list = []
	history_list2 = []
	perfect_threshold = 1.0
	loop_i = 0  # counts inner training iterations (used to throttle disk IO)
	if restarted_yet < 2:
		price_list_length = 10
	else:
		price_list_length = int(len(price_list)*0.5)
	while True:
		while True:
			loop_i += 1
			matched_patterns_count = 0
			list_of_ys = []
			list_of_ys_count = 0
			next_coin = 'no'
			all_current_patterns = []
			memory_or_history = []
			memory_weights = []

			high_memory_weights = []
			low_memory_weights = []
			final_moves = 0.0
			high_final_moves = 0.0
			low_final_moves = 0.0
			memory_indexes = []
			matches_yep = []
			flipped = 'no'
			last_minute = int(time.time()/60)
			overunder = 'nothing'
			overunder2 = 'nothing'
			list_of_ys = []
			all_predictions = []
			all_preds = []
			high_all_predictions = []
			high_all_preds = []
			low_all_predictions = []
			low_all_preds = []
			try:
				open_price_list2 = []
				open_price_list_index = 0
				while True:
					open_price_list2.append(open_price_list[open_price_list_index])
					open_price_list_index += 1
					if open_price_list_index >= price_list_length:
						break
					else:
						continue
			except:
				break
			low_all_preds = []
			try:
				price_list2 = []
				price_list_index = 0
				while True:
					price_list2.append(price_list[price_list_index])
					price_list_index += 1
					if price_list_index >= price_list_length:
						break
					else:
						continue
			except:
				break
			high_price_list2 = []
			high_price_list_index = 0
			while True:
				high_price_list2.append(high_price_list[high_price_list_index])
				high_price_list_index += 1
				if high_price_list_index >= price_list_length:
					break
				else:
					continue
			low_price_list2 = []
			low_price_list_index = 0
			while True:
				low_price_list2.append(low_price_list[low_price_list_index])
				low_price_list_index += 1
				if low_price_list_index >= price_list_length:
					break
				else:
					continue
			# Vectorized price change calculations
			if NUMPY_AVAILABLE:
				price_change_list = vectorized_price_changes(price_list2, open_price_list2).tolist()
				high_price_change_list = vectorized_price_changes(high_price_list2, open_price_list2).tolist()
				low_price_change_list = vectorized_price_changes(low_price_list2, open_price_list2).tolist()
			else:
				# Fallback to loop-based calculation
				index = 0
				price_change_list = []
				while True:
					price_change = 100*((price_list2[index]-open_price_list2[index])/open_price_list2[index])
					price_change_list.append(price_change)
					index += 1
					if index >= len(price_list2):
						break
				index = 0
				high_price_change_list = []
				while True:
					high_price_change = 100*((high_price_list2[index]-open_price_list2[index])/open_price_list2[index])
					high_price_change_list.append(high_price_change)
					index += 1
					if index >= len(price_list2):
						break
				index = 0
				low_price_change_list = []
				while True:
					low_price_change = 100*((low_price_list2[index]-open_price_list2[index])/open_price_list2[index])
					low_price_change_list.append(low_price_change)
					index += 1
					if index >= len(price_list2):
						break
			# Check stop signal occasionally (much less disk IO)
			if should_stop_training(loop_i):
				exited = 'yes'
				print('finished processing')
				file = open('trainer_last_start_time.txt','w+')
				file.write(str(start_time_yes))
				file.close()

				# Mark training finished for the GUI
				try:
					_trainer_finished_at = int(time.time())
					file = open('trainer_last_training_time.txt','w+')
					file.write(str(_trainer_finished_at))
					file.close()
				except:
					pass
				try:
					with open("trainer_status.json", "w", encoding="utf-8") as f:
						json.dump(
							{
								"coin": _arg_coin,
								"state": "FINISHED",
								"started_at": _trainer_started_at,
								"finished_at": _trainer_finished_at,
								"timestamp": _trainer_finished_at,
							},
							f,
						)
				except Exception:
					pass

				# Flush all memories for all timeframes before exit
				for tf in tf_choices:
					flush_memory(tf, force=True)
				print("All memories flushed to disk", flush=True)

				while True:
					continue
				the_big_index += 1
				restarted_yet = 0
				avg50 = []
				import sys
				import datetime
				import traceback
				import linecache
				import base64
				import calendar
				import hashlib
				import hmac
				from datetime import datetime
				sells_count = 0
				prediction_prices_avg_list = []
				pt_server = 'server'
				import psutil
				import logging
				list_len = 0
				restarting = 'no'
				in_trade = 'no'
				updowncount = 0
				updowncount1 = 0
				updowncount1_2 = 0
				updowncount1_3 = 0
				updowncount1_4 = 0
				high_var2 = 0.0
				low_var2 = 0.0
				last_flipped = 'no'
				starting_amounth02 = 100.0
				starting_amounth05 = 100.0
				starting_amounth10 = 100.0
				starting_amounth20 = 100.0
				starting_amounth50 = 100.0
				starting_amount = 100.0
				starting_amount1 = 100.0
				starting_amount1_2 = 100.0
				starting_amount1_3 = 100.0
				starting_amount1_4 = 100.0
				starting_amount2 = 100.0
				starting_amount2_2 = 100.0
				starting_amount2_3 = 100.0
				starting_amount2_4 = 100.0
				starting_amount3 = 100.0
				starting_amount3_2 = 100.0
				starting_amount3_3 = 100.0
				starting_amount3_4 = 100.0
				starting_amount4 = 100.0
				starting_amount4_2 = 100.0
				starting_amount4_3 = 100.0
				starting_amount4_4 = 100.0
				profit_list = []
				profit_list1 = []
				profit_list1_2 = []
				profit_list1_3 = []
				profit_list1_4 = []
				profit_list2 = []
				profit_list2_2 = []
				profit_list2_3 = []
				profit_list2_4 = []
				profit_list3 = []
				profit_list3_2 = []
				profit_list3_3 = []
				profit_list4 = []
				profit_list4_2 = []
				good_hits = []
				good_preds = []
				good_preds2 = []
				good_preds3 = []
				good_preds4 = []
				good_preds5 = []
				good_preds6 = []
				big_good_preds = []
				big_good_preds2 = []
				big_good_preds3 = []
				big_good_preds4 = []
				big_good_preds5 = []
				big_good_preds6 = []
				big_good_hits = []
				upordown = []
				upordown1 = []
				upordown1_2 = []
				upordown1_3 = []
				upordown1_4 = []
				upordown2 = []
				upordown2_2 = []
				upordown2_3 = []
				upordown2_4 = []
				upordown3 = []
				upordown3_2 = []
				upordown3_3 = []
				upordown3_4 = []
				upordown4 = []
				upordown4_2 = []
				upordown4_3 = []
				upordown4_4 = []
				upordown5 = []
				import json
				import uuid
				def PrintException():
					exc_type, exc_obj, tb = sys.exc_info()
					f = tb.tb_frame
					lineno = tb.tb_lineno
					filename = f.f_code.co_filename
					linecache.checkcache(filename)
					line = linecache.getline(filename, lineno, f.f_globals)
					print ('EXCEPTION IN (LINE {} "{}"): {}'.format(lineno, line.strip(), exc_obj))
				how_far_to_look_back = 100000
				list_len = 0
				if the_big_index >= len(tf_choices):
					if len(number_of_candles) == 1:
						print("Finished processing all timeframes (number_of_candles has only one entry). Exiting.")
						try:
							file = open('trainer_last_start_time.txt','w+')
							file.write(str(start_time_yes))
							file.close()
						except:
							pass

						# Mark training finished for the GUI
						try:
							_trainer_finished_at = int(time.time())
							file = open('trainer_last_training_time.txt','w+')
							file.write(str(_trainer_finished_at))
							file.close()
						except:
							pass
						try:
							with open("trainer_status.json", "w", encoding="utf-8") as f:
								json.dump(
									{
										"coin": _arg_coin,
										"state": "FINISHED",
										"started_at": _trainer_started_at,
										"finished_at": _trainer_finished_at,
										"timestamp": _trainer_finished_at,
									},
									f,
								)
						except Exception:
							pass

						sys.exit(0)
					else:
						the_big_index = 0
				else:
					pass

				break
			else:
				exited = 'no'
			perfect = []
			while True:
				try:
					print('\n\n\n\n')
					print(choice_index)
					print(restarted_yet)
					print(tf_list[restarted_yet])
					try:
						current_pattern_length = number_of_candles[number_of_candles_index]
						index = (len(price_change_list))-(number_of_candles[number_of_candles_index]-1)
						current_pattern = []
						history_pattern_start_index = (len(price_change_list))-((number_of_candles[number_of_candles_index]+candles_to_predict)*2)
						history_pattern_index = history_pattern_start_index
						while True:
							current_pattern.append(price_change_list[index])
							index += 1
							if len(current_pattern) >= (number_of_candles[number_of_candles_index]-1):
								break
							else:
								continue
					except:
						PrintException()
					try:
						high_current_pattern_length = number_of_candles[number_of_candles_index]
						index = (len(high_price_change_list))-(number_of_candles[number_of_candles_index]-1)
						high_current_pattern = []
						while True:
							high_current_pattern.append(high_price_change_list[index])
							index += 1
							if len(high_current_pattern) >= (number_of_candles[number_of_candles_index]-1):
								break
							else:
								continue
					except:
						PrintException()
					try:
						low_current_pattern_length = number_of_candles[number_of_candles_index]
						index = (len(low_price_change_list))-(number_of_candles[number_of_candles_index]-1)
						low_current_pattern = []
						while True:
							low_current_pattern.append(low_price_change_list[index])
							index += 1
							if len(low_current_pattern) >= (number_of_candles[number_of_candles_index]-1):
								break
							else:
								continue
					except:
						PrintException()
					history_diff = 1000000.0
					memory_diff = 1000000.0
					history_diffs = []
					memory_diffs = []
					if 1 == 1:
						try:
							# Use cached memory data
							_mem = load_memory(tf_choice)
							memory_list = _mem["memory_list"]
							weight_list = _mem["weight_list"]
							high_weight_list = _mem["high_weight_list"]
							low_weight_list = _mem["low_weight_list"]
							
							# Pre-parse memory patterns to NumPy arrays for faster matching
							# Limit memories to check in test mode
							max_memories_to_check = 10 if _test_mode else len(memory_list)
							memories_to_check = min(max_memories_to_check, len(memory_list))
							
							# Pre-parse current pattern
							if NUMPY_AVAILABLE:
								current_pattern_arr = np.array([float(x) for x in current_pattern], dtype=np.float64)
							else:
								current_pattern_arr = [float(x) for x in current_pattern]
							
							# Pre-parse all memory patterns
							parsed_memory_patterns = []
							for i in range(memories_to_check):
								if i < len(memory_list):
									parsed_pattern = parse_memory_to_array(memory_list[i])
									parsed_memory_patterns.append(parsed_pattern)
							
							mem_ind = 0
							diffs_list = []
							any_perfect = 'no'
							perfect_dexs = []
							perfect_diffs = []
							moves = []
							move_weights = []
							high_move_weights = []
							low_move_weights = []
							unweighted = []
							high_unweighted = []
							low_unweighted = []
							high_moves = []
							low_moves = []
							
							print(f"Checking {memories_to_check} memories for pattern matching...", flush=True)
							
							while mem_ind < memories_to_check:
								if mem_ind >= len(memory_list):
									break
								
								# Use pre-parsed pattern if available
								if NUMPY_AVAILABLE and mem_ind < len(parsed_memory_patterns):
									memory_pattern_arr = parsed_memory_patterns[mem_ind]
									if len(memory_pattern_arr) > 0 and len(current_pattern_arr) > 0:
										# Use optimized pattern difference calculation
										diff_avg = calculate_pattern_difference(current_pattern_arr, memory_pattern_arr)
									else:
										diff_avg = 1000000.0
								else:
									# Fallback to original calculation
									memory_pattern = memory_list[mem_ind].split('{}')[0].replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split(' ')
									checks = []
									check_dex = 0
									while check_dex < len(current_pattern):
										current_candle = float(current_pattern[check_dex])
										memory_candle = float(memory_pattern[check_dex])
										if current_candle + memory_candle == 0.0:
											difference = 0.0
										else:
											try:
												difference = abs((abs(current_candle-memory_candle)/((current_candle+memory_candle)/2))*100)
											except:
												difference = 0.0
										checks.append(difference)
										check_dex += 1
									diff_avg = sum(checks)/len(checks) if checks else 1000000.0
								
								diffs_list.append(diff_avg)
								
								if diff_avg <= perfect_threshold:
									any_perfect = 'yes'
									high_diff, low_diff = parse_memory_metadata(memory_list[mem_ind])
									high_diff = high_diff / 100
									low_diff = low_diff / 100
									
									# Extract move value
									memory_pattern = memory_list[mem_ind].split('{}')[0].replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split(' ')
									move_value = float(memory_pattern[len(memory_pattern)-1]) if len(memory_pattern) > 0 else 0.0
									
									unweighted.append(move_value)
									move_weights.append(float(weight_list[mem_ind]))
									high_move_weights.append(float(high_weight_list[mem_ind]))
									low_move_weights.append(float(low_weight_list[mem_ind]))
									high_unweighted.append(high_diff)
									low_unweighted.append(low_diff)
									moves.append(move_value * float(weight_list[mem_ind]))
									high_moves.append(high_diff * float(high_weight_list[mem_ind]))
									low_moves.append(low_diff * float(low_weight_list[mem_ind]))
									perfect_dexs.append(mem_ind)
									perfect_diffs.append(diff_avg)
									
									# Early exit: excellent match found
									if diff_avg < 0.01 and len(perfect_dexs) >= 1:
										print("✅ Excellent match found! Stopping memory check.", flush=True)
										break
								
								mem_ind += 1
							
							if any_perfect == 'no':
								memory_diff = min(diffs_list) if diffs_list else 1000000.0
								which_memory_index = diffs_list.index(memory_diff) if diffs_list else 0
								perfect.append('no')
								final_moves = 0.0
								high_final_moves = 0.0
								low_final_moves = 0.0
								new_memory = 'yes'
							else:
								try:
									final_moves = sum(moves)/len(moves) if moves else 0.0
									high_final_moves = sum(high_moves)/len(high_moves) if high_moves else 0.0
									low_final_moves = sum(low_moves)/len(low_moves) if low_moves else 0.0
								except:
									final_moves = 0.0
									high_final_moves = 0.0
									low_final_moves = 0.0
								which_memory_index = perfect_dexs[perfect_diffs.index(min(perfect_diffs))] if perfect_dexs else 0
								perfect.append('yes')
								print(f"Found {len(perfect_dexs)} matching memories", flush=True)
						except:
							PrintException()
							memory_list = []
							weight_list = []
							high_weight_list = []
							low_weight_list = []
							which_memory_index = 'no'
							perfect.append('no')
							diffs_list = []
							any_perfect = 'no'
							perfect_dexs = []
							perfect_diffs = []
							moves = []
							move_weights = []
							high_move_weights = []
							low_move_weights = []
							unweighted = []
							high_moves = []
							low_moves = []
							final_moves = 0.0
							high_final_moves = 0.0
							low_final_moves = 0.0
					else:
						pass
					all_current_patterns.append(current_pattern)
					if len(unweighted) > 20:
						if perfect_threshold < 0.1:
							perfect_threshold -= 0.001
						else:
							perfect_threshold -= 0.01
						if perfect_threshold < 0.0:
							perfect_threshold = 0.0
						else:
							pass
					else:
						if perfect_threshold < 0.1:
							perfect_threshold += 0.001
						else:
							perfect_threshold += 0.01
						if perfect_threshold > 100.0:
							perfect_threshold = 100.0
						else:
							pass
					write_threshold_sometimes(tf_choice, perfect_threshold, loop_i, every=200)

					try:
						index = 0
						current_pattern_length = number_of_candles[number_of_candles_index]
						index = (len(price_list2))-current_pattern_length
						current_pattern = []
						while True:
							current_pattern.append(price_list2[index])
							if len(current_pattern)>=number_of_candles[number_of_candles_index]:
								break
							else:
								index += 1
								if index >= len(price_list2):
									break
								else:
									continue	
					except:
						PrintException()
					if 1==1:
						while True:
							try:
								c_diff = final_moves/100
								high_diff = high_final_moves
								low_diff = low_final_moves
								prediction_prices = [current_pattern[len(current_pattern)-1]]
								high_prediction_prices = [current_pattern[len(current_pattern)-1]]
								low_prediction_prices = [current_pattern[len(current_pattern)-1]]
								start_price = current_pattern[len(current_pattern)-1]
								new_price = start_price+(start_price*c_diff)
								high_new_price = start_price+(start_price*high_diff)
								low_new_price = start_price+(start_price*low_diff)
								prediction_prices = [start_price,new_price]
								high_prediction_prices = [start_price,high_new_price]
								low_prediction_prices = [start_price,low_new_price]
							except:
								start_price = current_pattern[len(current_pattern)-1]
								new_price = start_price
								prediction_prices = [start_price,start_price]
								high_prediction_prices = [start_price,start_price]
								low_prediction_prices = [start_price,start_price]
							break
						index = len(current_pattern)-1
						index2 = 0
						all_preds.append(prediction_prices)
						high_all_preds.append(high_prediction_prices)
						low_all_preds.append(low_prediction_prices)
						overunder = 'within'
						all_predictions.append(prediction_prices)
						high_all_predictions.append(high_prediction_prices)
						low_all_predictions.append(low_prediction_prices)
						index = 0
						print(tf_choice)
						page_info = ''
						current_pattern_length = 3
						index = (len(price_list2)-1)-current_pattern_length
						current_pattern = []
						while True:
							current_pattern.append(price_list2[index])
							index += 1
							if index >= len(price_list2):
								break
							else:
								continue
						high_current_pattern_length = 3
						high_index = (len(high_price_list2)-1)-high_current_pattern_length
						high_current_pattern = []
						while True:
							high_current_pattern.append(high_price_list2[high_index])
							high_index += 1
							if high_index >= len(high_price_list2):
								break
							else:
								continue
						low_current_pattern_length = 3
						low_index = (len(low_price_list2)-1)-low_current_pattern_length
						low_current_pattern = []
						while True:
							low_current_pattern.append(low_price_list2[low_index])
							low_index += 1
							if low_index >= len(low_price_list2):
								break
							else:
								continue
						try:
							which_pattern_length = 0
							new_y = [start_price,new_price]
							high_new_y = [start_price,high_new_price]
							low_new_y = [start_price,low_new_price]
						except:
							PrintException()
							new_y = [current_pattern[len(current_pattern)-1],current_pattern[len(current_pattern)-1]]
							high_new_y = [current_pattern[len(current_pattern)-1],high_current_pattern[len(high_current_pattern)-1]]
							low_new_y = [current_pattern[len(current_pattern)-1],low_current_pattern[len(low_current_pattern)-1]]
					else:
						current_pattern_length = 3
						index = (len(price_list2))-current_pattern_length
						current_pattern = []
						while True:
							current_pattern.append(price_list2[index])
							index += 1
							if index >= len(price_list2):
								break
							else:
								continue
						high_current_pattern_length = 3
						high_index = (len(high_price_list2)-1)-high_current_pattern_length
						high_current_pattern = []
						while True:
							high_current_pattern.append(high_price_list2[high_index])
							high_index += 1
							if high_index >= len(high_price_list2):
								break
							else:
								continue
						low_current_pattern_length = 3
						low_index = (len(low_price_list2)-1)-low_current_pattern_length
						low_current_pattern = []
						while True:
							low_current_pattern.append(low_price_list2[low_index])
							low_index += 1
							if low_index >= len(low_price_list2):
								break
							else:
								continue
						new_y = [current_pattern[len(current_pattern)-1],current_pattern[len(current_pattern)-1]]
						number_of_candles_index += 1
						if number_of_candles_index >= len(number_of_candles):
							print("Processed all number_of_candles. Exiting.")
							sys.exit(0)
					perfect_yes = 'no'
					if 1==1:
						high_current_price = high_current_pattern[len(high_current_pattern)-1]
						low_current_price = low_current_pattern[len(low_current_pattern)-1]
						try:
							try:
								difference_of_actuals = last_actual-new_y[0]
								difference_of_last = last_actual-last_prediction
								percent_difference_of_actuals = ((new_y[0]-last_actual)/abs(last_actual))*100
								high_difference_of_actuals = last_actual-high_current_price
								high_percent_difference_of_actuals = ((high_current_price-last_actual)/abs(last_actual))*100
								low_difference_of_actuals = last_actual-low_current_price
								low_percent_difference_of_actuals = ((low_current_price-last_actual)/abs(last_actual))*100
								percent_difference_of_last = ((last_prediction-last_actual)/abs(last_actual))*100
								high_percent_difference_of_last = ((high_last_prediction-last_actual)/abs(last_actual))*100
								low_percent_difference_of_last = ((low_last_prediction-last_actual)/abs(last_actual))*100
								if in_trade == 'no':
									percent_for_no_sell = ((new_y[1]-last_actual)/abs(last_actual))*100
									og_actual = last_actual
									in_trade = 'yes'
								else:
									percent_for_no_sell = ((new_y[1]-og_actual)/abs(og_actual))*100
							except:
								difference_of_actuals = 0.0
								difference_of_last = 0.0
								percent_difference_of_actuals = 0.0
								percent_difference_of_last = 0.0
								high_difference_of_actuals = 0.0
								high_percent_difference_of_actuals = 0.0
								low_difference_of_actuals = 0.0
								low_percent_difference_of_actuals = 0.0
								high_percent_difference_of_last = 0.0
								low_percent_difference_of_last = 0.0
						except:
							PrintException()
						try:
							perdex = 0
							while True:
								if perfect[perdex] == 'yes':
									perfect_yes = 'yes'
									break
								else:
									perdex += 1
									if perdex >= len(perfect):                                                                        
										perfect_yes = 'no'
										break
									else:
										continue
							high_var = high_percent_difference_of_last
							low_var = low_percent_difference_of_last
							if last_flipped == 'no':
								if high_percent_difference_of_actuals >= high_var2+(high_var2*0.005) and percent_difference_of_actuals < high_var2:
									upordown3.append(1)
									upordown.append(1)
									upordown4.append(1)
									if len(upordown4) > 100:
										del upordown4[0]
									else:
										pass 
								elif low_percent_difference_of_actuals <= low_var2-(low_var2*0.005) and percent_difference_of_actuals > low_var2:
									upordown.append(1)
									upordown3.append(1)
									upordown4.append(1)
									if len(upordown4) > 100:
										del upordown4[0]
									else:
										pass  									
								elif high_percent_difference_of_actuals >= high_var2+(high_var2*0.005) and percent_difference_of_actuals > high_var2:
									upordown3.append(0)
									upordown2.append(0)
									upordown.append(0)
									upordown4.append(0)
									if len(upordown4) > 100:
										del upordown4[0]
									else:
										pass
								elif low_percent_difference_of_actuals <= low_var2-(low_var2*0.005) and percent_difference_of_actuals < low_var2:
									upordown3.append(0)
									upordown2.append(0)
									upordown.append(0)
									upordown4.append(0)
									if len(upordown4) > 100:
										del upordown4[0]
									else:
										pass  
								else:
									pass
							else:
								pass
							try:
								print('(Bounce Accuracy for last 100 Over Limit Candles): ' + format((sum(upordown4)/len(upordown4))*100,'.2f'))
							except:
								pass
							try:
								print('current candle: '+str(len(price_list2)))
							except:
								pass
							try:
								print('Total Candles: '+str(int(len(price_list))))
							except:
								pass
						except:
							PrintException()
					else:
						pass
					cc_on = 'no'
					try:
						long_trade = 'no'
						short_trade = 'no'
						last_moves = moves
						last_high_moves = high_moves
						last_low_moves = low_moves
						last_move_weights = move_weights
						last_high_move_weights = high_move_weights
						last_low_move_weights = low_move_weights
						last_perfect_dexs = perfect_dexs
						last_perfect_diffs = perfect_diffs
						percent_difference_of_now = ((new_y[1]-new_y[0])/abs(new_y[0]))*100
						high_percent_difference_of_now = ((high_new_y[1]-high_new_y[0])/abs(high_new_y[0]))*100
						low_percent_difference_of_now = ((low_new_y[1]-low_new_y[0])/abs(low_new_y[0]))*100
						high_var2 = high_percent_difference_of_now
						low_var2 = low_percent_difference_of_now
						var2 = percent_difference_of_now
						if flipped == 'yes':
							new1 = high_percent_difference_of_now
							high_percent_difference_of_now = low_percent_difference_of_now
							low_percent_difference_of_now = new1
						else:
							pass
					except:
						PrintException()
					last_actual = new_y[0]
					last_prediction = new_y[1]
					high_last_prediction = high_new_y[1]
					low_last_prediction = low_new_y[1]
					prediction_adjuster = 0.0
					prediction_expander2 = 1.5
					ended_on = number_of_candles_index
					next_coin = 'yes'
					profit_hit = 'no'
					long_profit = 0
					short_profit = 0
					"""
					expander_move = input('Expander good? yes or new number: ')
					if expander_move == 'yes':
						pass
					else:
						prediction_expander = expander_move
						continue
					"""
					last_flipped = flipped
					which_candle_of_the_prediction_index = 0
					if 1 == 1:
						current_pattern_ending = [current_pattern[len(current_pattern)-1]]
						while True:
							try:
								try:
									price_list_length += 1		
									which_candle_of_the_prediction_index += 1
									try:
										if len(price_list2)>=int(len(price_list)*0.25) and restarted_yet < 2:
											restarted_yet += 1
											restarting = 'yes'
											break
										else:
											restarting = 'no'
									except:
										restarting = 'no'
									if len(price_list2) == len(price_list):
										the_big_index += 1
										restarted_yet = 0
										print('restarting')
										restarting = 'yes'
										avg50 = []
										import sys
										import datetime
										import traceback
										import linecache
										import base64
										import calendar
										import hashlib
										import hmac
										from datetime import datetime
										sells_count = 0
										prediction_prices_avg_list = []
										pt_server = 'server'
										import psutil
										import logging
										list_len = 0
										in_trade = 'no'
										updowncount = 0
										updowncount1 = 0
										updowncount1_2 = 0
										updowncount1_3 = 0
										updowncount1_4 = 0
										high_var2 = 0.0
										low_var2 = 0.0
										last_flipped = 'no'
										starting_amounth02 = 100.0
										starting_amounth05 = 100.0
										starting_amounth10 = 100.0
										starting_amounth20 = 100.0
										starting_amounth50 = 100.0
										starting_amount = 100.0
										starting_amount1 = 100.0
										starting_amount1_2 = 100.0
										starting_amount1_3 = 100.0
										starting_amount1_4 = 100.0
										starting_amount2 = 100.0
										starting_amount2_2 = 100.0
										starting_amount2_3 = 100.0
										starting_amount2_4 = 100.0
										starting_amount3 = 100.0
										starting_amount3_2 = 100.0
										starting_amount3_3 = 100.0
										starting_amount3_4 = 100.0
										starting_amount4 = 100.0
										starting_amount4_2 = 100.0
										starting_amount4_3 = 100.0
										starting_amount4_4 = 100.0
										profit_list = []
										profit_list1 = []
										profit_list1_2 = []
										profit_list1_3 = []
										profit_list1_4 = []
										profit_list2 = []
										profit_list2_2 = []
										profit_list2_3 = []
										profit_list2_4 = []
										profit_list3 = []
										profit_list3_2 = []
										profit_list3_3 = []
										profit_list4 = []
										profit_list4_2 = []
										good_hits = []
										good_preds = []
										good_preds2 = []
										good_preds3 = []
										good_preds4 = []
										good_preds5 = []
										good_preds6 = []
										big_good_preds = []
										big_good_preds2 = []
										big_good_preds3 = []
										big_good_preds4 = []
										big_good_preds5 = []
										big_good_preds6 = []
										big_good_hits = []
										upordown = []
										upordown1 = []
										upordown1_2 = []
										upordown1_3 = []
										upordown1_4 = []
										upordown2 = []
										upordown2_2 = []
										upordown2_3 = []
										upordown2_4 = []
										upordown3 = []
										upordown3_2 = []
										upordown3_3 = []
										upordown3_4 = []
										upordown4 = []
										upordown4_2 = []
										upordown4_3 = []
										upordown4_4 = []
										upordown5 = []
										import json
										import uuid
										how_far_to_look_back = 100000
										list_len = 0
										print(the_big_index)
										print(len(tf_choices))
										if the_big_index >= len(tf_choices):
											if len(number_of_candles) == 1:
												print("Finished processing all timeframes (number_of_candles has only one entry). Exiting.")
												try:
													file = open('trainer_last_start_time.txt','w+')
													file.write(str(start_time_yes))
													file.close()
												except:
													pass

												# Mark training finished for the GUI
												try:
													_trainer_finished_at = int(time.time())
													file = open('trainer_last_training_time.txt','w+')
													file.write(str(_trainer_finished_at))
													file.close()
												except:
													pass
												try:
													with open("trainer_status.json", "w", encoding="utf-8") as f:
														json.dump(
															{
																"coin": _arg_coin,
																"state": "FINISHED",
																"started_at": _trainer_started_at,
																"finished_at": _trainer_finished_at,
																"timestamp": _trainer_finished_at,
															},
															f,
														)
												except Exception:
													pass

												# Final flush all memories before exit
												for tf in tf_choices:
													flush_memory(tf, force=True)
												print("All memories flushed to disk before exit", flush=True)

												sys.exit(0)
											else:
												the_big_index = 0
										else:
											pass
										break
									else:
										exited = 'no'
										try:
											price_list2 = []
											price_list_index = 0
											while True:
												price_list2.append(price_list[price_list_index])
												price_list_index += 1
												if len(price_list2) >= price_list_length:
													break
												else:
													continue
											high_price_list2 = []
											high_price_list_index = 0
											while True:
												high_price_list2.append(high_price_list[high_price_list_index])
												high_price_list_index += 1
												if high_price_list_index >= price_list_length:
													break
												else:
													continue
											low_price_list2 = []
											low_price_list_index = 0
											while True:
												low_price_list2.append(low_price_list[low_price_list_index])
												low_price_list_index += 1
												if low_price_list_index >= price_list_length:
													break
												else:
													continue
											price2 = price_list2[len(price_list2)-1]
											high_price2 = high_price_list2[len(high_price_list2)-1]
											low_price2 = low_price_list2[len(low_price_list2)-1]
											highlowind = 0
											this_differ = ((price2-new_y[1])/abs(new_y[1]))*100
											high_this_differ = ((high_price2-new_y[1])/abs(new_y[1]))*100
											low_this_differ = ((low_price2-new_y[1])/abs(new_y[1]))*100
											this_diff = ((price2-new_y[0])/abs(new_y[0]))*100
											high_this_diff = ((high_price2-new_y[0])/abs(new_y[0]))*100
											low_this_diff = ((low_price2-new_y[0])/abs(new_y[0]))*100
											difference_list = []
											list_of_predictions = all_predictions
											close_enough_counter = []
											which_pattern_length_index = 0								
											while True:
												current_prediction_price = all_predictions[highlowind][which_candle_of_the_prediction_index]
												high_current_prediction_price = high_all_predictions[highlowind][which_candle_of_the_prediction_index]
												low_current_prediction_price = low_all_predictions[highlowind][which_candle_of_the_prediction_index]
												perc_diff_now = ((current_prediction_price-new_y[0])/abs(new_y[0]))*100
												perc_diff_now_actual = ((price2-new_y[0])/abs(new_y[0]))*100
												high_perc_diff_now_actual = ((high_price2-new_y[0])/abs(new_y[0]))*100
												low_perc_diff_now_actual = ((low_price2-new_y[0])/abs(new_y[0]))*100
												try:
													difference = abs((abs(current_prediction_price-float(price2))/((current_prediction_price+float(price2))/2))*100)
												except:
													difference = 100.0
												try:
													direction = 'down'
													try:
														indy = 0
														while True:
															# Check bounds before accessing lists
															if indy >= len(moves) or indy >= len(high_moves) or indy >= len(low_moves) or indy >= len(unweighted):
																break
															new_memory = 'no'
															var3 = (moves[indy]*100)
															high_var3 = (high_moves[indy]*100)
															low_var3 = (low_moves[indy]*100)
															if high_perc_diff_now_actual > high_var3+(high_var3*0.1):
																high_new_weight = high_move_weights[indy] + 0.25
																if high_new_weight > 2.0:
																	high_new_weight = 2.0
																else:
																	pass
															elif high_perc_diff_now_actual < high_var3-(high_var3*0.1):
																high_new_weight = high_move_weights[indy] - 0.25
																if high_new_weight < 0.0:
																	high_new_weight = 0.0
																else:
																	pass
															else:
																high_new_weight = high_move_weights[indy]
															if low_perc_diff_now_actual < low_var3-(low_var3*0.1):
																low_new_weight = low_move_weights[indy] + 0.25
																if low_new_weight > 2.0:
																	low_new_weight = 2.0
																else:
																	pass
															elif low_perc_diff_now_actual > low_var3+(low_var3*0.1):
																low_new_weight = low_move_weights[indy] - 0.25
																if low_new_weight < 0.0:
																	low_new_weight = 0.0
																else:
																	pass
															else:
																low_new_weight = low_move_weights[indy]
															if perc_diff_now_actual > var3+(var3*0.1):
																new_weight = move_weights[indy] + 0.25
																if new_weight > 2.0:
																	new_weight = 2.0
																else:
																	pass
															elif perc_diff_now_actual < var3-(var3*0.1):
																new_weight = move_weights[indy] - 0.25
																if new_weight < (0.0-2.0):
																	new_weight = (0.0-2.0)
																else:
																	pass
															else:
																new_weight = move_weights[indy]
															
															# Direct assignment optimization (O(1) instead of O(n))
															idx = perfect_dexs[indy]
															if idx < len(weight_list):
																weight_list[idx] = new_weight
															if idx < len(high_weight_list):
																high_weight_list[idx] = high_new_weight
															if idx < len(low_weight_list):
																low_weight_list[idx] = low_new_weight

															# mark dirty (we will flush in batches)
															_mem = load_memory(tf_choice)
															_mem["dirty"] = True

															# occasional batch flush
															if loop_i % 200 == 0:
																flush_memory(tf_choice)

															indy += 1
															if indy >= len(unweighted):
																break
															else:
																pass
													except:
														PrintException()
														all_current_patterns[highlowind].append(this_diff)

														# build the same memory entry format, but store in RAM
														mem_entry = str(all_current_patterns[highlowind]).replace("'","").replace(',','').replace('"','').replace(']','').replace('[','')+'{}'+str(high_this_diff)+'{}'+str(low_this_diff)

														_mem = load_memory(tf_choice)
														_mem["memory_list"].append(mem_entry)
														_mem["weight_list"].append('1.0')
														_mem["high_weight_list"].append('1.0')
														_mem["low_weight_list"].append('1.0')
														_mem["dirty"] = True

														# occasional batch flush
														if loop_i % 200 == 0:
															flush_memory(tf_choice)

												except:
													PrintException()
													pass										
												highlowind += 1
												if highlowind >= len(all_predictions):
													break
												else:
													continue
										except:
											PrintException()
											while True:
												continue
									if which_candle_of_the_prediction_index >= candles_to_predict:
										break
									else:
										continue
								except:
									PrintException()
									while True:
										continue
							except:
								PrintException()
								while True:
									continue
					else:
						pass
					coin_choice_index += 1
					history_list = []
					price_change_list = []
					current_pattern = []
					break
				except:
					PrintException()
					while True:
						continue
			if restarting == 'yes':
				break
			else:
				continue
		if restarting == 'yes':
			break
		else:
			continue
tf_choices = ['1hour', '2hour', '4hour', '8hour', '12hour', '1day', '1week']
tf_minutes = [60, 120, 240, 480, 720, 1440, 10080]
# --- GUI HUB INPUT (NO PROMPTS) ---
# Usage: python pt_trainer.py BTC [reprocess_yes|reprocess_no]
_arg_coin = "BTC"

try:
	if len(sys.argv) > 1 and str(sys.argv[1]).strip():
		_arg_coin = str(sys.argv[1]).strip().upper()
except Exception:
	_arg_coin = "BTC"

coin_choice = _arg_coin + '-USDT'

restart_processing = "yes"

# GUI reads this status file to know if this coin is TRAINING or FINISHED
_trainer_started_at = int(time.time())
try:
	with open("trainer_status.json", "w", encoding="utf-8") as f:
		json.dump(
			{
				"coin": _arg_coin,
				"state": "TRAINING",
				"started_at": _trainer_started_at,
				"timestamp": _trainer_started_at,
			},
			f,
		)
except Exception:
	pass


the_big_index = 0
while True:
	list_len = 0
	restarting = 'no'
	in_trade = 'no'
	updowncount = 0
	updowncount1 = 0
	updowncount1_2 = 0
	updowncount1_3 = 0
	updowncount1_4 = 0
	high_var2 = 0.0
	low_var2 = 0.0
	last_flipped = 'no'
	starting_amounth02 = 100.0
	starting_amounth05 = 100.0
	starting_amounth10 = 100.0
	starting_amounth20 = 100.0
	starting_amounth50 = 100.0
	starting_amount = 100.0
	starting_amount1 = 100.0
	starting_amount1_2 = 100.0
	starting_amount1_3 = 100.0
	starting_amount1_4 = 100.0
	starting_amount2 = 100.0
	starting_amount2_2 = 100.0
	starting_amount2_3 = 100.0
	starting_amount2_4 = 100.0
	starting_amount3 = 100.0
	starting_amount3_2 = 100.0
	starting_amount3_3 = 100.0
	starting_amount3_4 = 100.0
	starting_amount4 = 100.0
	starting_amount4_2 = 100.0
	starting_amount4_3 = 100.0
	starting_amount4_4 = 100.0
	profit_list = []
	profit_list1 = []
	profit_list1_2 = []
	profit_list1_3 = []
	profit_list1_4 = []
	profit_list2 = []
	profit_list2_2 = []
	profit_list2_3 = []
	profit_list2_4 = []
	profit_list3 = []
	profit_list3_2 = []
	profit_list3_3 = []
	profit_list4 = []
	profit_list4_2 = []
	good_hits = []
	good_preds = []
	good_preds2 = []
	good_preds3 = []
	good_preds4 = []
	good_preds5 = []
	good_preds6 = []
	big_good_preds = []
	big_good_preds2 = []
	big_good_preds3 = []
	big_good_preds4 = []
	big_good_preds5 = []
	big_good_preds6 = []
	big_good_hits = []
	upordown = []
	upordown1 = []
	upordown1_2 = []
	upordown1_3 = []
	upordown1_4 = []
	upordown2 = []
	upordown2_2 = []
	upordown2_3 = []
	upordown2_4 = []
	upordown3 = []
	upordown3_2 = []
	upordown3_3 = []
	upordown3_4 = []
	upordown4 = []
	upordown4_2 = []
	upordown4_3 = []
	upordown4_4 = []
	upordown5 = []
	tf_choice = tf_choices[the_big_index]
	_mem = load_memory(tf_choice)
	memory_list = _mem["memory_list"]
	weight_list = _mem["weight_list"]
	high_weight_list = _mem["high_weight_list"]
	low_weight_list = _mem["low_weight_list"]
	no_list = 'no' if len(memory_list) > 0 else 'yes'

	tf_list = ['1hour',tf_choice,tf_choice]
	choice_index = tf_choices.index(tf_choice)
	minutes_list = [60,tf_minutes[choice_index],tf_minutes[choice_index]]
	if restarted_yet < 2:
		timeframe = tf_list[restarted_yet]#droplet setting (create list for all timeframes)
		timeframe_minutes = minutes_list[restarted_yet]#droplet setting (create list for all timeframe_minutes)
	else:
		timeframe = tf_list[2]#droplet setting (create list for all timeframes)
		timeframe_minutes = minutes_list[2]#droplet setting (create list for all timeframe_minutes)
	start_time = int(time.time())
	restarting = 'no'
	success_rate = 85
	volume_success_rate = 60
	candles_to_predict = 1#droplet setting (Max is half of number_of_candles)(Min is 2)
	max_difference = .5
	preferred_difference = .4 #droplet setting (max profit_margin) (Min 0.01)
	min_good_matches = 1#droplet setting (Max 100) (Min 4)
	max_good_matches = 1#droplet setting (Max 100) (Min is min_good_matches)
	prediction_expander = 1.33
	prediction_expander2 = 1.5
	prediction_adjuster = 0.0
	diff_avg_setting = 0.01
	min_success_rate = 90
	histories = 'off'
	coin_choice_index = 0
	list_of_ys_count = 0
	last_difference_between = 0.0
	history_list = []
	history_list2 = []
	len_avg = []
	list_len = 0
	start_time = int(time.time())
	start_time_yes = start_time
	if 'n' in restart_processing.lower():
		try:
			file = open('trainer_last_start_time.txt','r')
			last_start_time = int(file.read())
			file.close()
		except:
			last_start_time = 0.0
	else:
		last_start_time = 0.0
	end_time = int(start_time-((1500*timeframe_minutes)*60))
	perc_comp = format((len(history_list2)/how_far_to_look_back)*100,'.2f')
	last_perc_comp = perc_comp+'kjfjakjdakd'
	while True:
		time.sleep(.5)
		try:
			history = str(market.get_kline(coin_choice,timeframe,startAt=end_time,endAt=start_time)).replace(']]','], ').replace('[[','[').split('], [')
		except Exception as e:
			PrintException()
			time.sleep(3.5)
			continue
		index = 0
		while True:
			history_list.append(history[index])
			index += 1
			if index >= len(history):
				break
			else:
				continue
		perc_comp = format((len(history_list)/how_far_to_look_back)*100,'.2f')
		print('gathering history')
		current_change = len(history_list)-list_len	
		try:
			print('\n\n\n\n')
			print(current_change)
			if current_change < 1000:
				break
			else:
				pass
		except:
			PrintException()
			pass
		len_avg.append(current_change)
		list_len = len(history_list)
		last_perc_comp = perc_comp
		start_time = end_time
		end_time = int(start_time-((1500*timeframe_minutes)*60))
		print(last_start_time)
		print(start_time)
		print(end_time)
		print('\n')
		if start_time <= last_start_time:
			break
		else:
			continue
	if timeframe == '1day' or timeframe == '1week':
		if restarted_yet == 0:
			index = int(len(history_list)/2)
		else:
			index = 1
	else:
		index = int(len(history_list)/2)
	price_list = []
	high_price_list = []
	low_price_list = []
	open_price_list = []
	volume_list = []
	minutes_passed = 0
	try:
		while True:
			working_minute = str(history_list[index]).replace('"','').replace("'","").split(", ")
			try:
				if index == 1:
					current_tf_time = float(working_minute[0].replace('[',''))
					last_tf_time = current_tf_time
				else:
					pass
				candle_time = float(working_minute[0].replace('[',''))
				openPrice = float(working_minute[1])                
				closePrice = float(working_minute[2])
				highPrice = float(working_minute[3])
				lowPrice = float(working_minute[4])
				open_price_list.append(openPrice)
				price_list.append(closePrice)
				high_price_list.append(highPrice)
				low_price_list.append(lowPrice)
				index += 1
				if index >= len(history_list):
					break
				else:
					continue
			except:
				PrintException()
				index += 1
				if index >= len(history_list):
					break
				else:
					continue
		open_price_list.reverse()
		price_list.reverse()
		high_price_list.reverse()
		low_price_list.reverse()
		ticker_data = str(market.get_ticker(coin_choice)).replace('"','').replace("'","").replace("[","").replace("{","").replace("]","").replace("}","").replace(",","").lower().split(' ')
		price = float(ticker_data[ticker_data.index('price:')+1])
	except:
		PrintException()
	history_list = []
	history_list2 = []
	perfect_threshold = 1.0
	loop_i = 0  # counts inner training iterations (used to throttle disk IO)
	if restarted_yet < 2:
		price_list_length = 10
	else:
		price_list_length = int(len(price_list)*0.5)
	while True:
		while True:
			loop_i += 1
			matched_patterns_count = 0
			list_of_ys = []
			list_of_ys_count = 0
			next_coin = 'no'
			all_current_patterns = []
			memory_or_history = []
			memory_weights = []

			high_memory_weights = []
			low_memory_weights = []
			final_moves = 0.0
			high_final_moves = 0.0
			low_final_moves = 0.0
			memory_indexes = []
			matches_yep = []
			flipped = 'no'
			last_minute = int(time.time()/60)
			overunder = 'nothing'
			overunder2 = 'nothing'
			list_of_ys = []
			all_predictions = []
			all_preds = []
			high_all_predictions = []
			high_all_preds = []
			low_all_predictions = []
			low_all_preds = []
			try:
				open_price_list2 = []
				open_price_list_index = 0
				while True:
					open_price_list2.append(open_price_list[open_price_list_index])
					open_price_list_index += 1
					if open_price_list_index >= price_list_length:
						break
					else:
						continue
			except:
				break
			low_all_preds = []
			try:
				price_list2 = []
				price_list_index = 0
				while True:
					price_list2.append(price_list[price_list_index])
					price_list_index += 1
					if price_list_index >= price_list_length:
						break
					else:
						continue
			except:
				break
			high_price_list2 = []
			high_price_list_index = 0
			while True:
				high_price_list2.append(high_price_list[high_price_list_index])
				high_price_list_index += 1
				if high_price_list_index >= price_list_length:
					break
				else:
					continue
			low_price_list2 = []
			low_price_list_index = 0
			while True:
				low_price_list2.append(low_price_list[low_price_list_index])
				low_price_list_index += 1
				if low_price_list_index >= price_list_length:
					break
				else:
					continue
			index = 0
			index2 = index+1
			price_change_list = []
			while True:
				price_change = 100*((price_list2[index]-open_price_list2[index])/open_price_list2[index])
				price_change_list.append(price_change)
				index += 1
				if index >= len(price_list2):
					break
				else:
					continue
			index = 0
			index2 = index+1
			high_price_change_list = []
			while True:
				high_price_change = 100*((high_price_list2[index]-open_price_list2[index])/open_price_list2[index])
				high_price_change_list.append(high_price_change)
				index += 1
				if index >= len(price_list2):
					break
				else:
					continue
			index = 0
			index2 = index+1
			low_price_change_list = []
			while True:
				low_price_change = 100*((low_price_list2[index]-open_price_list2[index])/open_price_list2[index])
				low_price_change_list.append(low_price_change)
				index += 1
				if index >= len(price_list2):
					break
				else:
					continue
			# Check stop signal occasionally (much less disk IO)
			if should_stop_training(loop_i):
				exited = 'yes'
				print('finished processing')
				file = open('trainer_last_start_time.txt','w+')
				file.write(str(start_time_yes))
				file.close()

				# Mark training finished for the GUI
				try:
					_trainer_finished_at = int(time.time())
					file = open('trainer_last_training_time.txt','w+')
					file.write(str(_trainer_finished_at))
					file.close()
				except:
					pass
				try:
					with open("trainer_status.json", "w", encoding="utf-8") as f:
						json.dump(
							{
								"coin": _arg_coin,
								"state": "FINISHED",
								"started_at": _trainer_started_at,
								"finished_at": _trainer_finished_at,
								"timestamp": _trainer_finished_at,
							},
							f,
						)
				except Exception:
					pass

				# Flush any cached memory/weights before we spin
				flush_memory(tf_choice, force=True)

				sys.exit(0)

				the_big_index += 1
				restarted_yet = 0
				avg50 = []
				import sys
				import datetime
				import traceback
				import linecache
				import base64
				import calendar
				import hashlib
				import hmac
				from datetime import datetime
				sells_count = 0
				prediction_prices_avg_list = []
				pt_server = 'server'
				import psutil
				import logging
				list_len = 0
				restarting = 'no'
				in_trade = 'no'
				updowncount = 0
				updowncount1 = 0
				updowncount1_2 = 0
				updowncount1_3 = 0
				updowncount1_4 = 0
				high_var2 = 0.0
				low_var2 = 0.0
				last_flipped = 'no'
				starting_amounth02 = 100.0
				starting_amounth05 = 100.0
				starting_amounth10 = 100.0
				starting_amounth20 = 100.0
				starting_amounth50 = 100.0
				starting_amount = 100.0
				starting_amount1 = 100.0
				starting_amount1_2 = 100.0
				starting_amount1_3 = 100.0
				starting_amount1_4 = 100.0
				starting_amount2 = 100.0
				starting_amount2_2 = 100.0
				starting_amount2_3 = 100.0
				starting_amount2_4 = 100.0
				starting_amount3 = 100.0
				starting_amount3_2 = 100.0
				starting_amount3_3 = 100.0
				starting_amount3_4 = 100.0
				starting_amount4 = 100.0
				starting_amount4_2 = 100.0
				starting_amount4_3 = 100.0
				starting_amount4_4 = 100.0
				profit_list = []
				profit_list1 = []
				profit_list1_2 = []
				profit_list1_3 = []
				profit_list1_4 = []
				profit_list2 = []
				profit_list2_2 = []
				profit_list2_3 = []
				profit_list2_4 = []
				profit_list3 = []
				profit_list3_2 = []
				profit_list3_3 = []
				profit_list4 = []
				profit_list4_2 = []
				good_hits = []
				good_preds = []
				good_preds2 = []
				good_preds3 = []
				good_preds4 = []
				good_preds5 = []
				good_preds6 = []
				big_good_preds = []
				big_good_preds2 = []
				big_good_preds3 = []
				big_good_preds4 = []
				big_good_preds5 = []
				big_good_preds6 = []
				big_good_hits = []
				upordown = []
				upordown1 = []
				upordown1_2 = []
				upordown1_3 = []
				upordown1_4 = []
				upordown2 = []
				upordown2_2 = []
				upordown2_3 = []
				upordown2_4 = []
				upordown3 = []
				upordown3_2 = []
				upordown3_3 = []
				upordown3_4 = []
				upordown4 = []
				upordown4_2 = []
				upordown4_3 = []
				upordown4_4 = []
				upordown5 = []
				import json
				import uuid
				how_far_to_look_back = 100000
				list_len = 0
				if the_big_index >= len(tf_choices):
					if len(number_of_candles) == 1:
						print("Finished processing all timeframes (number_of_candles has only one entry). Exiting.")
						try:
							file = open('trainer_last_start_time.txt','w+')
							file.write(str(start_time_yes))
							file.close()
						except:
							pass

						# Mark training finished for the GUI
						try:
							_trainer_finished_at = int(time.time())
							file = open('trainer_last_training_time.txt','w+')
							file.write(str(_trainer_finished_at))
							file.close()
						except:
							pass
						try:
							with open("trainer_status.json", "w", encoding="utf-8") as f:
								json.dump(
									{
										"coin": _arg_coin,
										"state": "FINISHED",
										"started_at": _trainer_started_at,
										"finished_at": _trainer_finished_at,
										"timestamp": _trainer_finished_at,
									},
									f,
								)
						except Exception:
							pass

						sys.exit(0)
					else:
						the_big_index = 0
				else:
					pass

				break
			else:
				exited = 'no'
			perfect = []
			while True:
				try:
					print('\n\n\n\n')
					print(choice_index)
					print(restarted_yet)
					print(tf_list[restarted_yet])
					try:
						current_pattern_length = number_of_candles[number_of_candles_index]
						index = (len(price_change_list))-(number_of_candles[number_of_candles_index]-1)
						current_pattern = []
						history_pattern_start_index = (len(price_change_list))-((number_of_candles[number_of_candles_index]+candles_to_predict)*2)
						history_pattern_index = history_pattern_start_index
						while True:
							current_pattern.append(price_change_list[index])
							index += 1
							if len(current_pattern) >= (number_of_candles[number_of_candles_index]-1):
								break
							else:
								continue
					except:
						PrintException()
					try:
						high_current_pattern_length = number_of_candles[number_of_candles_index]
						index = (len(high_price_change_list))-(number_of_candles[number_of_candles_index]-1)
						high_current_pattern = []
						while True:
							high_current_pattern.append(high_price_change_list[index])
							index += 1
							if len(high_current_pattern) >= (number_of_candles[number_of_candles_index]-1):
								break
							else:
								continue
					except:
						PrintException()
					try:
						low_current_pattern_length = number_of_candles[number_of_candles_index]
						index = (len(low_price_change_list))-(number_of_candles[number_of_candles_index]-1)
						low_current_pattern = []
						while True:
							low_current_pattern.append(low_price_change_list[index])
							index += 1
							if len(low_current_pattern) >= (number_of_candles[number_of_candles_index]-1):
								break
							else:
								continue
					except:
						PrintException()
					history_diff = 1000000.0
					memory_diff = 1000000.0
					history_diffs = []
					memory_diffs = []
					if 1 == 1:
						try:
							file = open('memories_'+tf_choice+'.txt','r')
							memory_list = file.read().replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split('~')
							file.close()
							file = open('memory_weights_'+tf_choice+'.txt','r')
							weight_list = file.read().replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split(' ')
							file.close()							
							file = open('memory_weights_high_'+tf_choice+'.txt','r')
							high_weight_list = file.read().replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split(' ')
							file.close()
							file = open('memory_weights_low_'+tf_choice+'.txt','r')
							low_weight_list = file.read().replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split(' ')
							file.close()
							mem_ind = 0
							diffs_list = []
							any_perfect = 'no'
							perfect_dexs = []
							perfect_diffs = []
							moves = []
							move_weights = []
							high_move_weights = []
							low_move_weights = []
							unweighted = []
							high_unweighted = []
							low_unweighted = []
							high_moves = []
							low_moves = []
							while True:
								memory_pattern = memory_list[mem_ind].split('{}')[0].replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split(' ')
								avgs = []
								checks = []
								check_dex = 0
								while True:
									current_candle = float(current_pattern[check_dex])
									memory_candle = float(memory_pattern[check_dex])
									if current_candle + memory_candle == 0.0:
										difference = 0.0
									else:
										try:
											difference = abs((abs(current_candle-memory_candle)/((current_candle+memory_candle)/2))*100)
										except:
											difference = 0.0
									checks.append(difference)
									check_dex += 1
									if check_dex >= len(current_pattern):
										break
									else:
										continue
								diff_avg = sum(checks)/len(checks)
								if diff_avg <= perfect_threshold:
									any_perfect = 'yes'
									high_diff = float(memory_list[mem_ind].split('{}')[1].replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').replace(' ',''))/100
									low_diff = float(memory_list[mem_ind].split('{}')[2].replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').replace(' ',''))/100
									unweighted.append(float(memory_pattern[len(memory_pattern)-1]))
									move_weights.append(float(weight_list[mem_ind]))
									high_move_weights.append(float(high_weight_list[mem_ind]))
									low_move_weights.append(float(low_weight_list[mem_ind]))
									high_unweighted.append(high_diff)
									low_unweighted.append(low_diff)
									moves.append(float(memory_pattern[len(memory_pattern)-1])*float(weight_list[mem_ind]))
									high_moves.append(high_diff*float(high_weight_list[mem_ind]))
									low_moves.append(low_diff*float(low_weight_list[mem_ind]))
									perfect_dexs.append(mem_ind)
									perfect_diffs.append(diff_avg)
								else:
									pass
								diffs_list.append(diff_avg)
								mem_ind += 1
								if mem_ind >= len(memory_list):
									if any_perfect == 'no':
										memory_diff = min(diffs_list)
										which_memory_index = diffs_list.index(memory_diff)
										perfect.append('no')
										final_moves = 0.0
										high_final_moves = 0.0
										low_final_moves = 0.0
										new_memory = 'yes'
									else:
										try:
											final_moves = sum(moves)/len(moves)
											high_final_moves = sum(high_moves)/len(high_moves)
											low_final_moves = sum(low_moves)/len(low_moves)
										except:
											final_moves = 0.0
											high_final_moves = 0.0
											low_final_moves = 0.0
										which_memory_index = perfect_dexs[perfect_diffs.index(min(perfect_diffs))]
										perfect.append('yes')
									break
								else:
									continue
						except:
							PrintException()
							memory_list = []
							weight_list = []
							high_weight_list = []
							low_weight_list = []
							which_memory_index = 'no'
							perfect.append('no')
							diffs_list = []
							any_perfect = 'no'
							perfect_dexs = []
							perfect_diffs = []
							moves = []
							move_weights = []
							high_move_weights = []
							low_move_weights = []
							unweighted = []
							high_moves = []
							low_moves = []
							final_moves = 0.0
							high_final_moves = 0.0
							low_final_moves = 0.0
					else:
						pass
					all_current_patterns.append(current_pattern)
					if len(unweighted) > 20:
						if perfect_threshold < 0.1:
							perfect_threshold -= 0.001
						else:
							perfect_threshold -= 0.01
						if perfect_threshold < 0.0:
							perfect_threshold = 0.0
						else:
							pass
					else:
						if perfect_threshold < 0.1:
							perfect_threshold += 0.001
						else:
							perfect_threshold += 0.01
						if perfect_threshold > 100.0:
							perfect_threshold = 100.0
						else:
							pass
					write_threshold_sometimes(tf_choice, perfect_threshold, loop_i, every=200)

					try:
						index = 0
						current_pattern_length = number_of_candles[number_of_candles_index]
						index = (len(price_list2))-current_pattern_length
						current_pattern = []
						while True:
							current_pattern.append(price_list2[index])
							if len(current_pattern)>=number_of_candles[number_of_candles_index]:
								break
							else:
								index += 1
								if index >= len(price_list2):
									break
								else:
									continue	
					except:
						PrintException()
					if 1==1:
						while True:
							try:
								c_diff = final_moves/100
								high_diff = high_final_moves
								low_diff = low_final_moves
								prediction_prices = [current_pattern[len(current_pattern)-1]]
								high_prediction_prices = [current_pattern[len(current_pattern)-1]]
								low_prediction_prices = [current_pattern[len(current_pattern)-1]]
								start_price = current_pattern[len(current_pattern)-1]
								new_price = start_price+(start_price*c_diff)
								high_new_price = start_price+(start_price*high_diff)
								low_new_price = start_price+(start_price*low_diff)
								prediction_prices = [start_price,new_price]
								high_prediction_prices = [start_price,high_new_price]
								low_prediction_prices = [start_price,low_new_price]
							except:
								start_price = current_pattern[len(current_pattern)-1]
								new_price = start_price
								prediction_prices = [start_price,start_price]
								high_prediction_prices = [start_price,start_price]
								low_prediction_prices = [start_price,start_price]
							break
						index = len(current_pattern)-1
						index2 = 0
						all_preds.append(prediction_prices)
						high_all_preds.append(high_prediction_prices)
						low_all_preds.append(low_prediction_prices)
						overunder = 'within'
						all_predictions.append(prediction_prices)
						high_all_predictions.append(high_prediction_prices)
						low_all_predictions.append(low_prediction_prices)
						index = 0
						print(tf_choice)
						page_info = ''
						current_pattern_length = 3
						index = (len(price_list2)-1)-current_pattern_length
						current_pattern = []
						while True:
							current_pattern.append(price_list2[index])
							index += 1
							if index >= len(price_list2):
								break
							else:
								continue
						high_current_pattern_length = 3
						high_index = (len(high_price_list2)-1)-high_current_pattern_length
						high_current_pattern = []
						while True:
							high_current_pattern.append(high_price_list2[high_index])
							high_index += 1
							if high_index >= len(high_price_list2):
								break
							else:
								continue
						low_current_pattern_length = 3
						low_index = (len(low_price_list2)-1)-low_current_pattern_length
						low_current_pattern = []
						while True:
							low_current_pattern.append(low_price_list2[low_index])
							low_index += 1
							if low_index >= len(low_price_list2):
								break
							else:
								continue
						try:
							which_pattern_length = 0
							new_y = [start_price,new_price]
							high_new_y = [start_price,high_new_price]
							low_new_y = [start_price,low_new_price]
						except:
							PrintException()
							new_y = [current_pattern[len(current_pattern)-1],current_pattern[len(current_pattern)-1]]
							high_new_y = [current_pattern[len(current_pattern)-1],high_current_pattern[len(high_current_pattern)-1]]
							low_new_y = [current_pattern[len(current_pattern)-1],low_current_pattern[len(low_current_pattern)-1]]
					else:
						current_pattern_length = 3
						index = (len(price_list2))-current_pattern_length
						current_pattern = []
						while True:
							current_pattern.append(price_list2[index])
							index += 1
							if index >= len(price_list2):
								break
							else:
								continue
						high_current_pattern_length = 3
						high_index = (len(high_price_list2)-1)-high_current_pattern_length
						high_current_pattern = []
						while True:
							high_current_pattern.append(high_price_list2[high_index])
							high_index += 1
							if high_index >= len(high_price_list2):
								break
							else:
								continue
						low_current_pattern_length = 3
						low_index = (len(low_price_list2)-1)-low_current_pattern_length
						low_current_pattern = []
						while True:
							low_current_pattern.append(low_price_list2[low_index])
							low_index += 1
							if low_index >= len(low_price_list2):
								break
							else:
								continue
						new_y = [current_pattern[len(current_pattern)-1],current_pattern[len(current_pattern)-1]]
						number_of_candles_index += 1
						if number_of_candles_index >= len(number_of_candles):
							print("Processed all number_of_candles. Exiting.")
							sys.exit(0)
					perfect_yes = 'no'
					if 1==1:
						high_current_price = high_current_pattern[len(high_current_pattern)-1]
						low_current_price = low_current_pattern[len(low_current_pattern)-1]
						try:
							try:
								difference_of_actuals = last_actual-new_y[0]
								difference_of_last = last_actual-last_prediction
								percent_difference_of_actuals = ((new_y[0]-last_actual)/abs(last_actual))*100
								high_difference_of_actuals = last_actual-high_current_price
								high_percent_difference_of_actuals = ((high_current_price-last_actual)/abs(last_actual))*100
								low_difference_of_actuals = last_actual-low_current_price
								low_percent_difference_of_actuals = ((low_current_price-last_actual)/abs(last_actual))*100
								percent_difference_of_last = ((last_prediction-last_actual)/abs(last_actual))*100
								high_percent_difference_of_last = ((high_last_prediction-last_actual)/abs(last_actual))*100
								low_percent_difference_of_last = ((low_last_prediction-last_actual)/abs(last_actual))*100
								if in_trade == 'no':
									percent_for_no_sell = ((new_y[1]-last_actual)/abs(last_actual))*100
									og_actual = last_actual
									in_trade = 'yes'
								else:
									percent_for_no_sell = ((new_y[1]-og_actual)/abs(og_actual))*100
							except:
								difference_of_actuals = 0.0
								difference_of_last = 0.0
								percent_difference_of_actuals = 0.0
								percent_difference_of_last = 0.0
								high_difference_of_actuals = 0.0
								high_percent_difference_of_actuals = 0.0
								low_difference_of_actuals = 0.0
								low_percent_difference_of_actuals = 0.0
								high_percent_difference_of_last = 0.0
								low_percent_difference_of_last = 0.0
						except:
							PrintException()
						try:
							perdex = 0
							while True:
								if perfect[perdex] == 'yes':
									perfect_yes = 'yes'
									break
								else:
									perdex += 1
									if perdex >= len(perfect):                                                                        
										perfect_yes = 'no'
										break
									else:
										continue
							high_var = high_percent_difference_of_last
							low_var = low_percent_difference_of_last
							if last_flipped == 'no':
								if high_percent_difference_of_actuals >= high_var2+(high_var2*0.005) and percent_difference_of_actuals < high_var2:
									upordown3.append(1)
									upordown.append(1)
									upordown4.append(1)
									if len(upordown4) > 100:
										del upordown4[0]
									else:
										pass 
								elif low_percent_difference_of_actuals <= low_var2-(low_var2*0.005) and percent_difference_of_actuals > low_var2:
									upordown.append(1)
									upordown3.append(1)
									upordown4.append(1)
									if len(upordown4) > 100:
										del upordown4[0]
									else:
										pass  									
								elif high_percent_difference_of_actuals >= high_var2+(high_var2*0.005) and percent_difference_of_actuals > high_var2:
									upordown3.append(0)
									upordown2.append(0)
									upordown.append(0)
									upordown4.append(0)
									if len(upordown4) > 100:
										del upordown4[0]
									else:
										pass
								elif low_percent_difference_of_actuals <= low_var2-(low_var2*0.005) and percent_difference_of_actuals < low_var2:
									upordown3.append(0)
									upordown2.append(0)
									upordown.append(0)
									upordown4.append(0)
									if len(upordown4) > 100:
										del upordown4[0]
									else:
										pass  
								else:
									pass
							else:
								pass
							try:
								print('(Bounce Accuracy for last 100 Over Limit Candles): ' + format((sum(upordown4)/len(upordown4))*100,'.2f'))
							except:
								pass
							try:
								print('current candle: '+str(len(price_list2)))
							except:
								pass
							try:
								print('Total Candles: '+str(int(len(price_list))))
							except:
								pass
						except:
							PrintException()
					else:
						pass
					cc_on = 'no'
					try:
						long_trade = 'no'
						short_trade = 'no'
						last_moves = moves
						last_high_moves = high_moves
						last_low_moves = low_moves
						last_move_weights = move_weights
						last_high_move_weights = high_move_weights
						last_low_move_weights = low_move_weights
						last_perfect_dexs = perfect_dexs
						last_perfect_diffs = perfect_diffs
						percent_difference_of_now = ((new_y[1]-new_y[0])/abs(new_y[0]))*100
						high_percent_difference_of_now = ((high_new_y[1]-high_new_y[0])/abs(high_new_y[0]))*100
						low_percent_difference_of_now = ((low_new_y[1]-low_new_y[0])/abs(low_new_y[0]))*100
						high_var2 = high_percent_difference_of_now
						low_var2 = low_percent_difference_of_now
						var2 = percent_difference_of_now
						if flipped == 'yes':
							new1 = high_percent_difference_of_now
							high_percent_difference_of_now = low_percent_difference_of_now
							low_percent_difference_of_now = new1
						else:
							pass
					except:
						PrintException()
					last_actual = new_y[0]
					last_prediction = new_y[1]
					high_last_prediction = high_new_y[1]
					low_last_prediction = low_new_y[1]
					prediction_adjuster = 0.0
					prediction_expander2 = 1.5
					ended_on = number_of_candles_index
					next_coin = 'yes'
					profit_hit = 'no'
					long_profit = 0
					short_profit = 0
					"""
					expander_move = input('Expander good? yes or new number: ')
					if expander_move == 'yes':
						pass
					else:
						prediction_expander = expander_move
						continue
					"""
					last_flipped = flipped
					which_candle_of_the_prediction_index = 0
					if 1 == 1:
						current_pattern_ending = [current_pattern[len(current_pattern)-1]]
						while True:
							try:
								try:
									price_list_length += 1		
									which_candle_of_the_prediction_index += 1
									try:
										if len(price_list2)>=int(len(price_list)*0.25) and restarted_yet < 2:
											restarted_yet += 1
											restarting = 'yes'
											break
										else:
											restarting = 'no'
									except:
										restarting = 'no'
									if len(price_list2) == len(price_list):
										the_big_index += 1
										restarted_yet = 0
										print('restarting')
										restarting = 'yes'
										avg50 = []
										import sys
										import datetime
										import traceback
										import linecache
										import base64
										import calendar
										import hashlib
										import hmac
										from datetime import datetime
										sells_count = 0
										prediction_prices_avg_list = []
										pt_server = 'server'
										import psutil
										import logging
										list_len = 0
										in_trade = 'no'
										updowncount = 0
										updowncount1 = 0
										updowncount1_2 = 0
										updowncount1_3 = 0
										updowncount1_4 = 0
										high_var2 = 0.0
										low_var2 = 0.0
										last_flipped = 'no'
										starting_amounth02 = 100.0
										starting_amounth05 = 100.0
										starting_amounth10 = 100.0
										starting_amounth20 = 100.0
										starting_amounth50 = 100.0
										starting_amount = 100.0
										starting_amount1 = 100.0
										starting_amount1_2 = 100.0
										starting_amount1_3 = 100.0
										starting_amount1_4 = 100.0
										starting_amount2 = 100.0
										starting_amount2_2 = 100.0
										starting_amount2_3 = 100.0
										starting_amount2_4 = 100.0
										starting_amount3 = 100.0
										starting_amount3_2 = 100.0
										starting_amount3_3 = 100.0
										starting_amount3_4 = 100.0
										starting_amount4 = 100.0
										starting_amount4_2 = 100.0
										starting_amount4_3 = 100.0
										starting_amount4_4 = 100.0
										profit_list = []
										profit_list1 = []
										profit_list1_2 = []
										profit_list1_3 = []
										profit_list1_4 = []
										profit_list2 = []
										profit_list2_2 = []
										profit_list2_3 = []
										profit_list2_4 = []
										profit_list3 = []
										profit_list3_2 = []
										profit_list3_3 = []
										profit_list4 = []
										profit_list4_2 = []
										good_hits = []
										good_preds = []
										good_preds2 = []
										good_preds3 = []
										good_preds4 = []
										good_preds5 = []
										good_preds6 = []
										big_good_preds = []
										big_good_preds2 = []
										big_good_preds3 = []
										big_good_preds4 = []
										big_good_preds5 = []
										big_good_preds6 = []
										big_good_hits = []
										upordown = []
										upordown1 = []
										upordown1_2 = []
										upordown1_3 = []
										upordown1_4 = []
										upordown2 = []
										upordown2_2 = []
										upordown2_3 = []
										upordown2_4 = []
										upordown3 = []
										upordown3_2 = []
										upordown3_3 = []
										upordown3_4 = []
										upordown4 = []
										upordown4_2 = []
										upordown4_3 = []
										upordown4_4 = []
										upordown5 = []
										import json
										import uuid
										how_far_to_look_back = 100000
										list_len = 0
										print(the_big_index)
										print(len(tf_choices))
										if the_big_index >= len(tf_choices):
											if len(number_of_candles) == 1:
												print("Finished processing all timeframes (number_of_candles has only one entry). Exiting.")
												try:
													file = open('trainer_last_start_time.txt','w+')
													file.write(str(start_time_yes))
													file.close()
												except:
													pass

												# Mark training finished for the GUI
												try:
													_trainer_finished_at = int(time.time())
													file = open('trainer_last_training_time.txt','w+')
													file.write(str(_trainer_finished_at))
													file.close()
												except:
													pass
												try:
													with open("trainer_status.json", "w", encoding="utf-8") as f:
														json.dump(
															{
																"coin": _arg_coin,
																"state": "FINISHED",
																"started_at": _trainer_started_at,
																"finished_at": _trainer_finished_at,
																"timestamp": _trainer_finished_at,
															},
															f,
														)
												except Exception:
													pass

												sys.exit(0)
											else:
												the_big_index = 0
										else:
											pass
										break
									else:
										exited = 'no'
										try:
											price_list2 = []
											price_list_index = 0
											while True:
												price_list2.append(price_list[price_list_index])
												price_list_index += 1
												if len(price_list2) >= price_list_length:
													break
												else:
													continue
											high_price_list2 = []
											high_price_list_index = 0
											while True:
												high_price_list2.append(high_price_list[high_price_list_index])
												high_price_list_index += 1
												if high_price_list_index >= price_list_length:
													break
												else:
													continue
											low_price_list2 = []
											low_price_list_index = 0
											while True:
												low_price_list2.append(low_price_list[low_price_list_index])
												low_price_list_index += 1
												if low_price_list_index >= price_list_length:
													break
												else:
													continue
											price2 = price_list2[len(price_list2)-1]
											high_price2 = high_price_list2[len(high_price_list2)-1]
											low_price2 = low_price_list2[len(low_price_list2)-1]
											highlowind = 0
											this_differ = ((price2-new_y[1])/abs(new_y[1]))*100
											high_this_differ = ((high_price2-new_y[1])/abs(new_y[1]))*100
											low_this_differ = ((low_price2-new_y[1])/abs(new_y[1]))*100
											this_diff = ((price2-new_y[0])/abs(new_y[0]))*100
											high_this_diff = ((high_price2-new_y[0])/abs(new_y[0]))*100
											low_this_diff = ((low_price2-new_y[0])/abs(new_y[0]))*100
											difference_list = []
											list_of_predictions = all_predictions
											close_enough_counter = []
											which_pattern_length_index = 0								
											while True:
												current_prediction_price = all_predictions[highlowind][which_candle_of_the_prediction_index]
												high_current_prediction_price = high_all_predictions[highlowind][which_candle_of_the_prediction_index]
												low_current_prediction_price = low_all_predictions[highlowind][which_candle_of_the_prediction_index]
												perc_diff_now = ((current_prediction_price-new_y[0])/abs(new_y[0]))*100
												perc_diff_now_actual = ((price2-new_y[0])/abs(new_y[0]))*100
												high_perc_diff_now_actual = ((high_price2-new_y[0])/abs(new_y[0]))*100
												low_perc_diff_now_actual = ((low_price2-new_y[0])/abs(new_y[0]))*100
												try:
													difference = abs((abs(current_prediction_price-float(price2))/((current_prediction_price+float(price2))/2))*100)
												except:
													difference = 100.0
												try:
													direction = 'down'
													try:
														indy = 0
														while True:
															# Check bounds before accessing lists
															if indy >= len(moves) or indy >= len(high_moves) or indy >= len(low_moves) or indy >= len(unweighted):
																break
															new_memory = 'no'
															var3 = (moves[indy]*100)
															high_var3 = (high_moves[indy]*100)
															low_var3 = (low_moves[indy]*100)
															if high_perc_diff_now_actual > high_var3+(high_var3*0.1):
																high_new_weight = high_move_weights[indy] + 0.25
																if high_new_weight > 2.0:
																	high_new_weight = 2.0
																else:
																	pass
															elif high_perc_diff_now_actual < high_var3-(high_var3*0.1):
																high_new_weight = high_move_weights[indy] - 0.25
																if high_new_weight < 0.0:
																	high_new_weight = 0.0
																else:
																	pass
															else:
																high_new_weight = high_move_weights[indy]
															if low_perc_diff_now_actual < low_var3-(low_var3*0.1):
																low_new_weight = low_move_weights[indy] + 0.25
																if low_new_weight > 2.0:
																	low_new_weight = 2.0
																else:
																	pass
															elif low_perc_diff_now_actual > low_var3+(low_var3*0.1):
																low_new_weight = low_move_weights[indy] - 0.25
																if low_new_weight < 0.0:
																	low_new_weight = 0.0
																else:
																	pass
															else:
																low_new_weight = low_move_weights[indy]
															if perc_diff_now_actual > var3+(var3*0.1):
																new_weight = move_weights[indy] + 0.25
																if new_weight > 2.0:
																	new_weight = 2.0
																else:
																	pass
															elif perc_diff_now_actual < var3-(var3*0.1):
																new_weight = move_weights[indy] - 0.25
																if new_weight < (0.0-2.0):
																	new_weight = (0.0-2.0)
																else:
																	pass
															else:
																new_weight = move_weights[indy]
															del weight_list[perfect_dexs[indy]]
															weight_list.insert(perfect_dexs[indy],new_weight)
															del high_weight_list[perfect_dexs[indy]]
															high_weight_list.insert(perfect_dexs[indy],high_new_weight)
															del low_weight_list[perfect_dexs[indy]]
															low_weight_list.insert(perfect_dexs[indy],low_new_weight)

															# mark dirty (we will flush in batches)
															_mem = load_memory(tf_choice)
															_mem["dirty"] = True

															# occasional batch flush
															if loop_i % 200 == 0:
																flush_memory(tf_choice)

															indy += 1
															if indy >= len(unweighted):
																break
															else:
																pass
													except:
														PrintException()
														all_current_patterns[highlowind].append(this_diff)

														# build the same memory entry format, but store in RAM
														mem_entry = str(all_current_patterns[highlowind]).replace("'","").replace(',','').replace('"','').replace(']','').replace('[','')+'{}'+str(high_this_diff)+'{}'+str(low_this_diff)

														_mem = load_memory(tf_choice)
														_mem["memory_list"].append(mem_entry)
														_mem["weight_list"].append('1.0')
														_mem["high_weight_list"].append('1.0')
														_mem["low_weight_list"].append('1.0')
														_mem["dirty"] = True

														# occasional batch flush
														if loop_i % 200 == 0:
															flush_memory(tf_choice)

												except:
													PrintException()
													pass										
												highlowind += 1
												if highlowind >= len(all_predictions):
													break
												else:
													continue
										except SystemExit:
											raise
										except KeyboardInterrupt:
											raise
										except Exception:
											PrintException()
											break

									if which_candle_of_the_prediction_index >= candles_to_predict:
										break
									else:
										continue
								except SystemExit:
									raise
								except KeyboardInterrupt:
									raise
								except Exception:
									PrintException()
									break

							except SystemExit:
								raise
							except KeyboardInterrupt:
								raise
							except Exception:
								PrintException()
								break

					else:
						pass
					coin_choice_index += 1
					history_list = []
					price_change_list = []
					current_pattern = []
					break
				except SystemExit:
					raise
				except KeyboardInterrupt:
					raise
				except Exception:
					PrintException()
					break

			if restarting == 'yes':
				break
			else:
				continue
		if restarting == 'yes':
			break
		else:
			continue
