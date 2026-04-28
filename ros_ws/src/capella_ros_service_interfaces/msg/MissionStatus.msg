# if nav and carto off : mission_status = 0
# if carto on : mission_status = 1; 
# if nav on : mission_status = 2
# if infrared on : mission_status = 3
uint8 MISSION_STATUS_IDLE = 0
uint8 MISSION_STATUS_CARTO = 1
uint8 MISSION_STATUS_NAV = 2
uint8 MISSION_STATUS_INFRARED = 3

uint8 mission_status