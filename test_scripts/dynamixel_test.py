import dynamixel_sdk as dxl

# Control table address
ADDR_PRO_TORQUE_ENABLE      = 64
ADDR_PRO_LED_RED            = 65
ADDR_PRO_GOAL_POSITION      = 116
ADDR_PRO_PRESENT_POSITION   = 132

# Protocol version
PROTOCOL_VERSION            = 2.0

# Default setting
DXL_ID                      = 1                # Dynamixel ID: 1
BAUDRATE                    = 57600             # Dynamixel default baudrate : 57600
DEVICENAME                  = '/dev/ttyACM0'    # Check which port is being used on your controller

TORQUE_ENABLE               = 1                 # Value for enabling the torque
TORQUE_DISABLE              = 0                 # Value for disabling the torque
DXL_MINIMUM_POSITION_VALUE  = 100               # Dynamixel will rotate between this value
DXL_MAXIMUM_POSITION_VALUE  = 4000              # and this value
DXL_MOVING_STATUS_THRESHOLD = 20                # Dynamixel moving status threshold

portHandler = dxl.PortHandler(DEVICENAME)
packetHandler = dxl.PacketHandler(PROTOCOL_VERSION)

# Open port
if not portHandler.openPort():
    print("Failed to open the port")
    quit()

# Set port baudrate
if not portHandler.setBaudRate(BAUDRATE):
    print("Failed to change the baudrate")
    quit()

# Enable Dynamixel Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_PRO_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != dxl.COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel has been successfully connected")

# Close port
portHandler.closePort()
