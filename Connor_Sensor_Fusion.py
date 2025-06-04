from __future__ import print_function
from mbientlab.metawear import MetaWear, libmetawear, parse_value, create_voidp, create_voidp_int
from mbientlab.metawear.cbindings import *
from time import sleep
from threading import Event
import keyboard
import socket
import can
import struct
import threading
import time
import math
from adafruit_servokit import ServoKit
import matplotlib.pyplot as plt
import os
import json
import datetime
import pandas as pd
import joblib
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
import gym

# Dummy state space and action space
class DummyEnv:
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(3,), dtype=np.float32)
env = DummyEnv()


model = DDPG.load("ddpg_pid_model_untrained.zip")

kit = ServoKit(channels = 16)

# Scaling constants for MIT Control Mode
P_MIN, P_MAX = -12.5, 12.5      # Position range (radians)
V_MIN, V_MAX = -50.0, 50.0      # Velocity range (rad/s)
KP_MIN, KP_MAX = 0.0, 500.0     # Kp range
KD_MIN, KD_MAX = 0.0, 5.0       # Kd range
T_MIN, T_MAX = -18.0, 18.0      # Torque range (Nm)

# set up the can interface
bus = can.interface.Bus(channel='can0', bustype='socketcan', bitrate=1000000)
controller_id = 2
# Define Addresses
adr_motor1 = 1
adr_motor2 = 2

td = 0.000001
decay = 0.99

GAIN_FILE = "pid_gains.pkl"
if os.path.exists(GAIN_FILE):
    gains = joblib.load(GAIN_FILE)
else:
    gains = {"Kp": 1.25, "Ki": 1.0, "Kd": 0.05}

# Fuck Around And Find Out
kp = 80
kd = 1
moddeg = 10


# Socket IP Address & Port
IP_Address = "10.101.172.68"
IP_Port = 12345

plt.ion()
fig, ax = plt.subplots()
line1, = ax.plot([], [], label='Actual Position', color='blue')
line2, = ax.plot([], [], label='Desired Position', color='orange')
ax.set_xlabel("Time Step")
ax.set_ylabel("Angle(Degrees)")
ax.set_title("Real-Time Position")
ax.legend()
ax.grid(True)

x_data = []
actual_data = []
desired_data = []

# Define sensor classes

class PIDController:
    def __init__(self, Kp, Ki, Kd, dt, output_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt  # Time step

        self.integral = 0.00
        self.prev_error = 0.00
        self.output_limits = output_limits  # (min_output, max_output)

    def reset(self):
        self.integral = 0.00
        self.prev_error = 0.00

    def compute(self, setpoint, actual_position):
        error = setpoint - actual_position
        self.integral *= decay
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt

        output = (
            self.Kp * error +
            self.Ki * self.integral +
            self.Kd * derivative
        )

        # Optional output clamping
        min_out, max_out = self.output_limits
        if min_out is not None:
            output = max(min_out, output)
        if max_out is not None:
            output = min(max_out, output)

        self.prev_error = error
        return output, derivative

class Sensor:
    def __init__(self, name, mac):
        self.name = name
        self.mac = mac
        self.data = []

class State:
	
    def __init__(self, device, name):
        self.device = device
        self.samples = 0
        self.accCallback = FnVoid_VoidP_DataP(self.acc_data_handler)
        self.name = name
        self.dataw = 0
        self.datax = 0
        self.datay = 0
        self.dataz = 0
        self.inityaw = 0
        self.initroll = 0
        self.initpitch = 0
    
    def float_to_uint(value, value_min, value_max, bits):
        """
        Converts a float to an unsigned integer given range and number of bits.
        """
        span = value_max - value_min
        scaled = int((value - value_min) * ((1 << bits) - 1) / span)
        return max(0, min((1 << bits) - 1, scaled))  # Clamp to valid range
	# Define control function, runs everytime data is received
    
    def send_mit_control(bus, controller_id, position, velocity, kp, kd, torque):
        """
        Sends a command in MIT Control Mode.

        Args:
            bus: The CAN bus instance.
            controller_id: The ID of the motor controller.
            position: Desired position (radians).
            velocity: Desired velocity (rad/s).
            kp: Position control gain.
            kd: Velocity control gain.
            torque: Desired feedforward torque (Nm).
        """
        # Scale values to unsigned integers
        p_int = float_to_uint(position, P_MIN, P_MAX, 16)
        v_int = float_to_uint(velocity, V_MIN, V_MAX, 12)
        kp_int = float_to_uint(kp, KP_MIN, KP_MAX, 12)
        kd_int = float_to_uint(kd, KD_MIN, KD_MAX, 12)
        t_int = float_to_uint(torque, T_MIN, T_MAX, 12)
        
        # Pack data into 8-byte buffer
        data = bytearray(8)
        data[0] = (p_int >> 8) & 0xFF  # Position high byte
        data[1] = p_int & 0xFF         # Position low byte
        data[2] = (v_int >> 4) & 0xFF  # Velocity high byte
        data[3] = ((v_int & 0xF) << 4) | ((kp_int >> 8) & 0xF)  # Velocity low + Kp high
        data[4] = kp_int & 0xFF        # Kp low byte
        data[5] = (kd_int >> 4) & 0xFF # Kd high byte
        data[6] = ((kd_int & 0xF) << 4) | ((t_int >> 8) & 0xF)  # Kd low + Torque high
        data[7] = t_int & 0xFF         # Torque low byte

        # Send CAN message
        can_id = controller_id
        msg = can.Message(arbitration_id=can_id, data=data, is_extended_id=False)
        
        try:
            bus.send(msg)
            print(f"Sent: ID={hex(can_id)}, Data={data.hex()}")
        except can.CanError as e:
            print(f"Failed to send CAN message: {e}")

    def send_servo_position_command(bus, motor_id, position_deg):
    	CONTROL_MODE_POSITION = 4
    	can_id = (motor_id | int(CONTROL_MODE_POSITION) << 8)
    	position_int = int(position_deg * 100000)
    	data = position_int.to_bytes(4, byteorder='big', signed=True)

    	msg = can.Message(arbitration_id=can_id, data=data, is_extended_id=True)
    	try:
        	bus.send(msg)
        	print(f"Sent position: {position_deg}°, ID: {hex(can_id)}, data: {data.hex()}")
    	except can.CanError as e:
        	print(f"CAN send error: {e}")

	
    def acc_data_handler(self, ctx, data):
        acc_data = parse_value(data)
        self.samples += 1
        #print("%s -> %s" % (self.name, acc_data))
        self.dataw = acc_data.w
        self.datax = acc_data.x
        self.datay = acc_data.y
        self.dataz = acc_data.z
        
        # acc_data = [state.dataw, state.datax, state.datay, state.dataz]
        # message = f"{self.name} -> Quaternion: {acc_data}".encode('utf-8')
        # sock.sendto(message, (IP_Address, IP_Port))
        
        #send_mit_control(bus,2, acc_data.w, 0, 3, .5, 0)
        #time.sleep(1)
        
        #this sends the data wirelessly
        

# Define sensor MAC adresses and create empty states array

sensors = [
    Sensor('Healthy Lower Arm', "C5:29:60:28:D6:C2")
    ,Sensor('Healthy Upper Arm', "DB:70:F3:A7:38:91")
    ,Sensor('Exo Upper Arm', "C5:BC:37:39:50:35")
    ,Sensor('Exo Lower Arm', "C3:B5:15:C5:0E:F1")
]

states = []

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)

pid = PIDController(Kp=gains["Kp"], Ki=gains["Ki"], Kd=gains["Kd"], dt=.18, output_limits=(-90,90))  # Adjust gains as needed


def get_and_increment_trial_number():
    filename = "trial_counter.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = {}

    date_str = datetime.datetime.now().strftime("%m_%d_%Y")
    trial_number = data.get(date_str, 1)

    # Increment and save for next time
    data[date_str] = trial_number + 1
    with open(filename, "w") as f:
        json.dump(data, f)

    return date_str, trial_number

def save_integral(integral):
    with open("pid_integral.json", "w") as f:
        json.dump({"integral": integral}, f)

def load_integral():
    try:
        with open("pid_integral.json", "r") as f:
            return json.load(f).get("integral", 0.0)
    except FileNotFoundError:
        return 0.0

def save_sensor_data_log(log_data, date_str, trial_number):
    dfi = pd.DataFrame(log_data)
    filename = f"{date_str}_Trial_{trial_number}.xlsx"
    dfi.to_excel(filename, index=False)

def send_mit_control(bus, controller_id, position, velocity, kp, kd, torque):
        """
        Sends a command in MIT Control Mode.

        Args:
            bus: The CAN bus instance.
            controller_id: The ID of the motor controller.
            position: Desired position (radians).
            velocity: Desired velocity (rad/s).
            kp: Position control gain.
            kd: Velocity control gain.
            torque: Desired feedforward torque (Nm).
        """
        # Scale values to unsigned integers
        p_int = float_to_uint(position, P_MIN, P_MAX, 16)
        v_int = float_to_uint(velocity, V_MIN, V_MAX, 12)
        kp_int = float_to_uint(kp, KP_MIN, KP_MAX, 12)
        kd_int = float_to_uint(kd, KD_MIN, KD_MAX, 12)
        t_int = float_to_uint(torque, T_MIN, T_MAX, 12)
        
        # Pack data into 8-byte buffer
        data = bytearray(8)
        data[0] = (p_int >> 8) & 0xFF  # Position high byte
        data[1] = p_int & 0xFF         # Position low byte
        data[2] = (v_int >> 4) & 0xFF  # Velocity high byte
        data[3] = ((v_int & 0xF) << 4) | ((kp_int >> 8) & 0xF)  # Velocity low + Kp high
        data[4] = kp_int & 0xFF        # Kp low byte
        data[5] = (kd_int >> 4) & 0xFF # Kd high byte
        data[6] = ((kd_int & 0xF) << 4) | ((t_int >> 8) & 0xF)  # Kd low + Torque high
        data[7] = t_int & 0xFF         # Torque low byte

        # Send CAN message
        can_id = controller_id
        msg = can.Message(arbitration_id=can_id, data=data, is_extended_id=False)
        try:
            bus.send(msg)
            #print(f"Sent: ID={hex(can_id)}, Data={data.hex()}")
        except can.CanError as e:
            print(f"Failed to send CAN message: {e}")

# Connects to the sensors and start data aquisition

def send_servo_position_command(bus, motor_id, position_deg):
    CONTROL_MODE_POSITION = 4
    can_id = (motor_id | int(CONTROL_MODE_POSITION) << 8)
    position_deg = round(position_deg, 1)
    position_int = int(position_deg * 10000)
    data = position_int.to_bytes(4, byteorder='big', signed=True)

    msg = can.Message(arbitration_id=can_id, data=data, is_extended_id=True)
    try:
        bus.send(msg)
        print(f"Sent position: {position_deg}°, ID: {hex(can_id)}, data: {data.hex()}")
    except can.CanError as e:
        print(f"CAN send error: {e}")


def float_to_uint(value, value_min, value_max, bits):
        """
        Converts a float to an unsigned integer given range and number of bits.
        """
        span = value_max - value_min
        scaled = int((value - value_min) * ((1 << bits) - 1) / span)
        return max(0, min((1 << bits) - 1, scaled))  # Clamp to valid range

def connect_and_configure(sensor):
    device = MetaWear(sensor.mac)
    device.connect()
    print("Connected to " + sensor.name + " over " + ("USB" if device.usb.is_connected else "BLE"))
    state = State(device, sensor.name)

    libmetawear.mbl_mw_sensor_fusion_set_mode(device.board, SensorFusionMode.IMU_PLUS)
    libmetawear.mbl_mw_sensor_fusion_set_acc_range(device.board, SensorFusionAccRange._8G)
    libmetawear.mbl_mw_sensor_fusion_set_gyro_range(device.board, SensorFusionGyroRange._2000DPS)
    libmetawear.mbl_mw_sensor_fusion_write_config(device.board)

    signal = libmetawear.mbl_mw_sensor_fusion_get_data_signal(device.board, SensorFusionData.QUATERNION)
    libmetawear.mbl_mw_datasignal_subscribe(signal, None, state.accCallback)
    
    states.append(state)
    
    libmetawear.mbl_mw_sensor_fusion_enable_data(device.board, SensorFusionData.QUATERNION)
    libmetawear.mbl_mw_sensor_fusion_start(device.board)

# Disconnects sensors and preps them for next use

def stop_and_disconnect(state):
    try:
        device = state.device
        libmetawear.mbl_mw_debug_disconnect(device.board)
        print("Disconnected from " + state.name + " over " + ("USB" if device.usb.is_connected else "BLE"))
    except Exception as e:
        print(f"Skipping {state.name} due to error: {e}")

def initialize(bus, controller_id):
    can_id = controller_id
    data = bytearray(8)
    data[0] = 0xFF
    data[1] = 0xFF
    data[2] = 0xFF
    data[3] = 0xFF
    data[4] = 0xFF
    data[5] = 0xFF
    data[6] = 0xFF
    data[7] = 0XFC
    msg = can.Message(arbitration_id=can_id, data=data, is_extended_id=False)

    try:
        bus.send(msg)
        print(f"Sent: ID={hex(can_id)}, Data={data.hex()}")
    except can.CanError as e:
        print(f"Failed to send CAN message: {e}")

def Exit(bus, controller_id):
    can_id = controller_id
    data = bytearray(8)
    data[0] = 0xFF
    data[1] = 0xFF
    data[2] = 0xFF
    data[3] = 0xFF
    data[4] = 0xFF
    data[5] = 0xFF
    data[6] = 0xFF
    data[7] = 0XFD
    msg = can.Message(arbitration_id=can_id, data=data, is_extended_id=False)

    try:
        bus.send(msg)
        #print(f"Sent: ID={hex(can_id)}, Data={data.hex()}")
    except can.CanError as e:
        print(f"Failed to send CAN message: {e}")
        
def init_angl_zero():
    for state in states:
        w, x, y, z = state.dataw, state.datax, state.datay, state.dataz
        norm = math.sqrt(w**2 + x**2 + y**2 + z**2)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        state.inityaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
        state.initroll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
        state.initpitch = math.asin(2.0 * (w * y - z * x))

def yaw_filter(state, yaw):
    inityaw = state.inityaw
    yaw = yaw - inityaw
    if yaw < -math.pi:
        yaw = yaw + (2 * math.pi)
    elif yaw > math.pi:
        yaw = yaw - (2 * math.pi)
    else:
        yaw = yaw
    return yaw
    
def pitch_filter(state, pitch):
    initpitch = state.initpitch
    pitch = pitch - initpitch
    if pitch < -math.pi:
        pitch = pitch + (2 * math.pi)
    elif pitch > math.pi:
        pitch = pitch - (2 * math.pi)
    else:
        pitch = pitch
    return pitch
    
def roll_filter(state, roll):
    initroll = state.initroll
    roll = roll - initroll
    if roll < -math.pi:
        roll = roll + (2 * math.pi)
    elif roll > math.pi:
        roll = roll - (2 * math.pi)
    else:
        roll = roll
    return roll
    
def SetServo(angle):
    if angle > 10:
        angle = 10
    elif angle < -90:
        angle = -90
    else:
        angle = angle
        
    # angle = angle + 10
    kit.servo[0].angle = angle + 90
    
def setzero(bus, controller_id):
    can_id = controller_id
    data = bytearray(8)
    data[0] = 0xFF
    data[1] = 0xFF
    data[2] = 0xFF
    data[3] = 0xFF
    data[4] = 0xFF
    data[5] = 0xFF
    data[6] = 0xFF
    data[7] = 0XFE
    msg = can.Message(arbitration_id=can_id, data=data, is_extended_id=False)

    try:
        bus.send(msg)
        print(f"Sent: ID={hex(can_id)}, Data={data.hex()}")
    except can.CanError as e:
        print(f"Failed to send CAN message: {e}")

def set_servo_origin(bus, motor_id, origin_mode=1):
    """
    Sets the origin (current position = 0) for the CubeMars motor in Servo Mode.

    Args:
        bus: CAN bus interface
        motor_id: Motor CAN ID
    initroll = state.initroll
        origin_mode: 0 = temporary, 1 = permanent (default), 2 = restore default
    """
    CONTROL_MODE_ORIGIN = 5
    can_id = (motor_id | int(CONTROL_MODE_ORIGIN) << 8)
    data = bytearray([origin_mode])  # only one byte required

    msg = can.Message(arbitration_id=can_id, data=data, is_extended_id=True)
    try:
        bus.send(msg)
        print(f"Sent origin set command (mode {origin_mode}) to motor {motor_id}")
    except can.CanError as e:
        print(f"CAN send error: {e}")


def main():
    #initialize(bus, 1)
    #time.sleep(1)
    #initialize(bus, 2)
    #time.sleep(1)

    #send_mit_control(bus, 1, 0.00, 0.00, 0.00, 0.00, 0.00)
    #time.sleep(1)
    #send_mit_control(bus, 2, 0.00, 0.00, 0.00, 0.00, 0.00)
    #time.sleep(1)

    #send_servo_position_command(bus, 1, 0) -62.955710956268064
    #time.sleep(1)
    #send_servo_position_command(bus, 2, 0)
    #time.sleep(1)
	
    SetServo(0)
    pid.integral = load_integral()
    
    #states = []
    
    # while True:
        # if len(sensors) == len(states):
            # break
        # else:
    #try:
        # states = []
    for sensor in sensors:
        connect_and_configure(sensor)
    # except Exception as e:
        # print("failed to cennect to all sensors, retrying")
    
    

    
    time.sleep(1)
    
    init_angl_zero()
    
    set_servo_origin(bus, 1)
    time.sleep(1)
    set_servo_origin(bus, 2)
    time.sleep(.01)
    sensor_data_log = []
    log_data = []
	desired_buffer = []
	actual_buffer = []
	N = 50
	step_counter = 0
    print("Press 'q' to stop streaming and disconnect sensors")

    while True:
        if len(sensors) != len(states):
            break
        
        start_time = time.time()
        print(f"start time is {start_time}")    
        # Import Current Data
        upperArm = states[1]
        lowerArm = states[0]         
        exoUpper = states[2]
        exoLower = states[3]
    
        # Calculate Angle Values
        w, x, y, z = upperArm.dataw, upperArm.datax, upperArm.datay, upperArm.dataz
        norm = math.sqrt(w**2 + x**2 + y**2 + z**2)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        mtr2angl = math.asin(2.0 * (w * y - z * x))
        mtr2angl = pitch_filter(upperArm, mtr2angl)
        
        mtr1angl = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
        mtr1angl = roll_filter(upperArm, mtr1angl)
        
        wl, xl, yl, zl = lowerArm.dataw, lowerArm.datax, lowerArm.datay, lowerArm.dataz
        norm = math.sqrt(wl**2 + xl**2 + yl**2 + zl**2)
        wl, xl, yl, zl = wl / norm, xl / norm, yl / norm, zl / norm
        
        upperYaw = math.asin(2.0 * (w * y - z * x))
        upperYaw = pitch_filter(upperArm, upperYaw)
        
        lowerYaw = math.asin(2.0 * (wl * yl - zl * xl))
        lowerYaw = pitch_filter(lowerArm, lowerYaw)
        
        lowerRoll = math.atan2(2.0 * (wl * xl + yl * zl), 1.0 - 2.0 * (xl**2 + yl**2))
        lowerRoll = roll_filter(lowerArm, lowerRoll)
        
        elbowAngle = upperYaw - lowerYaw
        elbowAngle = elbowAngle*(180/math.pi)
        
        we, xe, ye, ze = exoUpper.dataw, exoUpper.datax, exoUpper.datay, exoUpper.dataz
        norm = math.sqrt(we**2 + xe**2 + ye**2 + ze**2)
        we, xe, ye, ze = we / norm, xe / norm, ye / norm, ze / norm
        mtr2anglexo = math.asin(2.0 * (we * ye - ze * xe))
        mtr2anglexo = pitch_filter(exoUpper, mtr2anglexo)
        
        mtr1anglexo = math.atan2(2.0 * (we * xe + ye * ze), 1.0 - 2.0 * (xe**2 + ye**2))
        mtr1anglexo = roll_filter(exoUpper, mtr1anglexo)
        
        wel, xel, yel, zel = exoLower.dataw, exoLower.datax, exoLower.datay, exoLower.dataz
        norm = math.sqrt(wel**2 + xel**2 + yel**2 + zel**2)
        wel, xel, yel, zel = wel / norm, xel / norm, yel / norm, zel / norm
        
        lowerYawexo = math.asin(2.0 * (wel * yel - zel * xel))
        lowerYawexo = pitch_filter(exoLower, lowerYawexo)
        
        lowerRollexo = math.atan2(2.0 * (wel * xel + yel * zel), 1.0 - 2.0 * (xel**2 + yel**2))
        lowerRollexo = roll_filter(exoLower, lowerRollexo)
        
        diff1 = (mtr1angl - mtr1anglexo) 
        diff2 = (mtr2angl - mtr2anglexo)
        
        modrad = (moddeg/180) * math.pi
        
        desired_position = elbowAngle
        actual_position = (-mtr2anglexo + lowerYawexo) * (180/math.pi)
        
        correction, derivative = pid.compute(desired_position, actual_position)
        SetServo(correction + actual_position)
        if step_counter > 0 and step_counter % N == 0:
        	# Calculate reward
        	error = np.array(desired_buffer) - np.array(actual_buffer)
        	rms_error = np.sqrt(np.mean(error**2))
        	overshoot = max(0, np.max(actual_buffer) - np.max(desired_buffer))
        	reward = -rms_error - 5 * overshoot

        	# Build state vector
        	state = np.array(
            	desired_buffer + actual_buffer + [pid.Kp, pid.Ki, pid.Kd],
            	dtype=np.float32
        	).reshape(1, -1)

        	# Train DDPG model with pseudo experience
        	action, _ = model.predict(state, deterministic=False)
        	model.replay_buffer.add(state, action, [reward], state, [1.0])
        	model.train(batch_size=1, gradient_steps=1)

        	# Apply bounded delta gains
        	delta_kp = float(np.clip(action[0][0], -0.05, 0.05))
        	delta_ki = float(np.clip(action[0][1], -0.005, 0.005))
        	delta_kd = float(np.clip(action[0][2], -0.0005, 0.0005))
        	pid.Kp = float(np.clip(pid.Kp + delta_kp, 0.0, 10.0))
        	pid.Ki = float(np.clip(pid.Ki + delta_ki, 0.0, 2.0))
        	pid.Kd = float(np.clip(pid.Kd + delta_kd, 0.0, 1.0))

        	# Clear buffers for next cycle
        	desired_buffer.clear()
        	actual_buffer.clear()
        
        #print(mtr2angl*(180/3.14))
        
        print(f"motor 1 angle = {mtr1angl}")
        print(f"motor 2 angle = {mtr2angl}")
        print(f"lower yaw = {desired_position}")
        print(f"motor 1 angle exo = {mtr1anglexo}")
        print(f"motor 2 angle exo = {mtr2anglexo}")
        print(f"lower yaw exo = {actual_position}")
        
        # Set Motor Angles
        #SetServo(elbowAngle)
        time.sleep(td)
        send_servo_position_command(bus, 1, (mtr2angl * 180 / math.pi) * 2.8)
        time.sleep(td)
        send_servo_position_command(bus, 2, (mtr1angl * 180 / math.pi) * 1.745)
        time.sleep(td)
        
        # Send Wireless Data To Matlab
        message = f"{'Healthy Lower Arm'} -> Quaternion: {[lowerYaw, lowerRoll]} / ".encode('utf-8')
        sock.sendto(message, (IP_Address, IP_Port))
        message = f"{'Healthy Upper Arm'} -> Quaternion: {[mtr2angl, mtr1angl - modrad]} / ".encode('utf-8')
        sock.sendto(message, (IP_Address, IP_Port))
        message = f"{'Exo Lower Arm'} -> Quaternion: {[lowerYawexo, lowerRollexo]} / ".encode('utf-8')
        sock.sendto(message, (IP_Address, IP_Port))
        message = f"{'Exo Upper Arm'} -> Quaternion: {[mtr2anglexo, mtr1anglexo]} / ".encode('utf-8')
        sock.sendto(message, (IP_Address, IP_Port))
        
        x_data.append(len(x_data))
        actual_data.append(-actual_position)
        desired_data.append(-desired_position)
        
        line1.set_data(x_data, actual_data)
        line2.set_data(x_data, desired_data)
        
        ax.relim()
        ax.autoscale_view()
        
        plt.draw()
        plt.pause(td)
        error = desired_position - actual_position
        sensor_data_log.append({
    	    "timestamp": time.time(),
    	    "error": error,
    	    "exo elbow": actual_position,
    	    "healthy elbow": desired_position,
    	    "healthy shoulder yaw": mtr2angl,
            "healthy shoulder pitch": mtr1angl,
            "exo shoulder yaw": mtr2anglexo,
            "exo shoulder pitch": mtr1anglexo
        })

        log_data.append({
    	    "error": error,
            "derivative": derivative,
            "integral": pid.integral,
            "Kp": pid.Kp,
            "Ki": pid.Ki,
            "Kd": pid.Kd
        })
        
        if keyboard.is_pressed('q'):
            print('Stopping streaming and disconnecting sensors')
            #print(upperArm.data)
            break
        sleep(td)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"end time is {end_time}")
        print(f"execution time is {execution_time} seconds")
	    step_counter += 1

    for state in states:
        stop_and_disconnect(state)
        
    time.sleep(1)
    send_servo_position_command(bus, 1, 0.00)
    time.sleep(1)
    send_servo_position_command(bus, 2, 0.00)
    time.sleep(1)
    Exit(bus, 1)
    time.sleep(1)
    Exit(bus, 2)
    
    SetServo(0)
    
    sock.close()

    print("Saving integral and data...")
	
    # Save PID integral
    save_integral(pid.integral)

    # Get current date and trial number
    date_str, trial_number = get_and_increment_trial_number()

    # Save sensor readings
    save_sensor_data_log(sensor_data_log, date_str, trial_number)
    
    df = pd.DataFrame(log_data)
    filename = f"system_data_Trial_{trial_number}.csv"
    df.to_csv(filename, index=False)

	joblib.dump({"Kp": pid.Kp, "Ki": pid.Ki, "Kd": pid.Kd}, GAIN_FILE)
	model.save("ddpg_pid_model_untrained")
	
    print(f"Trial saved as: {date_str}_Trial_{trial_number}.xlsx")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
