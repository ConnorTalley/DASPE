#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  SensorFix.py
#  
#  Copyright 2025  <atekeslab@raspberrypi>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
from __future__ import print_function
from mbientlab.metawear import MetaWear, libmetawear, parse_value, create_voidp, create_voidp_int
from mbientlab.metawear.cbindings import *
from time import sleep
from threading import Event


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

    def acc_data_handler(self, ctx, data):
        acc_data = parse_value(data)
        self.samples += 1
        #print("%s -> %s" % (self.name, acc_data))
        self.dataw = acc_data.w
        self.datax = acc_data.x
        self.datay = acc_data.y
        self.dataz = acc_data.z

sensors = [
    Sensor('Healthy Lower Arm', "C5:29:60:28:D6:C2")
    ,Sensor('Healthy Upper Arm', "DB:70:F3:A7:38:91")
    ,Sensor('Exo Upper Arm', "C5:BC:37:39:50:35")
    ,Sensor('Exo Lower Arm', "C3:B5:15:C5:0E:F1")
]


states = []

i = 0

def connect_and_configure(sensor):
    device = MetaWear(sensor.mac)
    device.connect()
    print("Connected to " + sensor.name + " over " + ("USB" if device.usb.is_connected else "BLE"))
    state = State(device, sensor.name)
    states.append(state)
    
def stop_and_disconnect(state):
    try:
        device = state.device
        libmetawear.mbl_mw_debug_disconnect(device.board)
        print("Disconnected from " + state.name + " over " + ("USB" if device.usb.is_connected else "BLE"))
    except Exception as e:
        print(f"Skipping {state.name} due to error: {e}")
        
for sensor in sensors:
    
    try:
        connect_and_configure(sensor)
        state = states[i]
        stop_and_disconnect(state)
        i = i+1
    except Exception as e:
        print(f"skipping {sensor.name} due to error: {e}")
    
        
