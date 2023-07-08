import time

import odrive
from odrive.enums import *
import gamepad

class Robot:
    def __init__(self,connect=True,odrv0=None):
        # physical
        self.wheel_diameter_cm = 17.13
        self.wheel_circ_cm = self.wheel_diameter_cm * 3.14159
        self.degperrot = 117 #fudgefactor to make 90deg rotations actually 90deg

        # choices
        self.vel_cm_per_sec = 100.0
        self.accel_cm_per_sec2 = 20.0
        self.decel_cm_per_sec2 = 5.0

        # setup
        self.is_setup=False

        if connect:
            self.connect()

    def setup(self):
        self.is_setup=True
        self.odrive_setup()
        time.sleep(4)
        self.motor_setup()
        time.sleep(1)
        self.motors_calibrate()
        time.sleep(10)
        self.motors_closed()

        
    def connect(self):
        self.odrv0=odrive.find_any()


    def odrive_setup(self):
        o = self.odrv0
        o.config.dc_bus_overvoltage_ramp_end = 43
        o.config.dc_bus_overvoltage_ramp_start = 37
        o.config.dc_bus_overvoltage_trip_level = 48
        o.config.dc_bus_undervoltage_trip_level = 32
        o.config.dc_max_negative_current = -4
        o.config.dc_max_positive_current = 50
        o.config.max_regen_current = 3
        o.config.brake_resistance = 2
        o.config.enable_brake_resistor = True
        try:
            o.save_configuration()
        except:
            print("reconnecting")
            self.connect()

    
    def motor_setup(self):
        o = self.odrv0

        vel_limit_rps = self.vel_cm_per_sec / self.wheel_circ_cm
        accel_limit_rps2 = self.accel_cm_per_sec2 / self.wheel_circ_cm
        decel_limit_rps2 = self.decel_cm_per_sec2 / self.wheel_circ_cm

        for x in [o.axis0, o.axis1]:
            x.encoder.config.cpr = 3200
            x.motor.config.torque_constant = 8.27 / 8.75
            x.motor.config.pole_pairs = 15

            x.encoder.config.bandwidth = 1000
            

            #x.motor.config.motor_type = odrive.enums.MOTOR_TYPE_GIMBAL #gimbal
            #x.motor.config.calibration_current = 14.0  # VOLTS in gimbal mode
            #x.motor.config.current_lim = 24.0          # VOLTS in gimbal mode

            x.motor.config.motor_type = odrive.enums.MOTOR_TYPE_HIGH_CURRENT
         
            x.motor.config.requested_current_range = 25
            x.motor.config.calibration_current = 5
            x.motor.config.current_lim = 13.0          

            x.motor.config.resistance_calib_max_voltage = 14
            
            x.motor.config.torque_lim = 13.0

            x.motor.config.motor_type = odrive.enums.MOTOR_TYPE_GIMBAL #gimbal
            x.motor.config.calibration_current = 14.0  # VOLTS in gimbal mode
            x.motor.config.current_lim = 24.0          # VOLTS in gimbal mode
            
            x.motor.config.current_control_bandwidth = 100
            x.controller.config.pos_gain = 50 #50
            x.controller.config.vel_gain = .5 #.5
            x.controller.config.vel_integrator_gain = 10 #.5 in pos mode
            x.controller.config.vel_limit = 2.0
            x.controller.config.vel_limit_tolerance = 1.2

            x.trap_traj.config.accel_limit = accel_limit_rps2
            x.trap_traj.config.decel_limit = decel_limit_rps2
            x.trap_traj.config.vel_limit = vel_limit_rps

            #x.controller.config.control_mode = odrive.enums.CONTROL_MODE_POSITION_CONTROL
            #x.controller.config.input_mode = odrive.enums.INPUT_MODE_TRAP_TRAJ
            x.controller.config.control_mode = odrive.enums.CONTROL_MODE_VELOCITY_CONTROL
            x.controller.config.input_mode=odrive.enums.INPUT_MODE_PASSTHROUGH
            #x.controller.config.enable_torque_mode_vel_limit=False

            #x.controller.config.vel_ramp_rate = 0.5
            x.controller.input_vel = 0

    
    def motors_calibrate(self):
        o = self.odrv0
        o.axis0.requested_state = odrive.enums.AXIS_STATE_FULL_CALIBRATION_SEQUENCE
        o.axis1.requested_state = odrive.enums.AXIS_STATE_FULL_CALIBRATION_SEQUENCE

    
    def motors_closed(self):
        o = self.odrv0
        o.axis0.requested_state = odrive.enums.AXIS_STATE_CLOSED_LOOP_CONTROL
        o.axis1.requested_state = odrive.enums.AXIS_STATE_CLOSED_LOOP_CONTROL

    def motors_idle(self):
        o = self.odrv0
        o.axis0.requested_state = odrive.enums.AXIS_STATE_IDLE
        o.axis1.requested_state = odrive.enums.AXIS_STATE_IDLE
        
    def tune(self,p,v,i):
        o = self.odrv0
        o.axis0.controller.config.pos_gain=p
        o.axis1.controller.config.pos_gain=p
        o.axis0.controller.config.vel_gain=v
        o.axis1.controller.config.vel_gain=v
        o.axis0.controller.config.vel_integrator_gain=i
        o.axis1.controller.config.vel_integrator_gain=i

    def wheel_vels(self, vels_norm=None,top_speed=1):
        if vels_norm is None:
            vels_norm={'left':0,'right':0}

        o = self.odrv0
        o.axis0.controller.input_vel = vels_norm['right'] * top_speed
        o.axis1.controller.input_vel = -vels_norm['left'] * top_speed
    
    def robot_move(self, cm_left, cm_right):
        o = self.odrv0
        o.axis0.controller.move_incremental(cm_right / self.wheel_circ_cm, False)
        o.axis1.controller.move_incremental(-cm_left / self.wheel_circ_cm, False)

    
    def robot_spin(self, angle):
        o = self.odrv0
        deg = self.degperrot / 360
        o.axis0.controller.move_incremental(-(deg * angle) / self.wheel_circ_cm, False)
        o.axis1.controller.move_incremental(-(deg * angle) / self.wheel_circ_cm, False)

    def gamepad_control(self):
        top_speed=.5 #2 is limit of wheels,but accelleration too dangerous
        pad = gamepad.Gamepad()
    
        while 1:
            time.sleep(.1)
            vels_norm=pad.diffDrive()
            self.wheel_vels({'left':vels_norm['left'],'right':vels_norm['right']},top_speed)

        self.wheel_vels({'left':0,'right':0},top_speed)
        
            
        

    # def drive(self):
    #     if not self.is_setup:
    #         self.setup()
        
    #     old_settings = termios.tcgetattr(sys.stdin)
    #     try:
    #         tty.setcbreak(sys.stdin.fileno())
    #         print("drive with wsad. press q to quit")
    #         while True:
    #             key = sys.stdin.read(1)

    #             if key == 'q':
    #                 break
    #             elif key == 'w':
    #                 self.robot_move(30, 30)
    #             elif key == 's':
    #                 self.robot_move(-30, -30)
    #             elif key == 'a':
    #                 self.robot_spin(-45)
    #             elif key == 'd':
    #                 self.robot_spin(45)
    #             else:
    #                 print("Invalid command. Please try again.")
    #     finally:
    #         termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        
robot = Robot()
#robot.drive()
