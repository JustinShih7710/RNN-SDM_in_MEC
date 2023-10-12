import math
from numba import jit
@jit
def path_loss(D_Cell_UE):
    pL=15.3+37.6*math.log10(D_Cell_UE)
    return pL+30

@jit
def Power_received(pL):
    AntennaGain_Cell=44
    AntennaGain_UE=30
    Power_ConSumption=23
    P_received = 10**((AntennaGain_Cell+AntennaGain_UE+Power_ConSumption-pL)/10)
    return P_received
@jit
def SINR_UE(P_received):
    P_received_sum=0
    for n in P_received:
        P_received_sum = P_received_sum + n
    SINR_UE_value = P_received_sum/(10**(-17.4))
    return SINR_UE_value
@jit
def SINR_UE(P_received):
    SINR_UE_value = P_received/(10**(-17.4))
    return SINR_UE_value

def CQI_Table(SINR):
    if SINR <= -6.9360:
        codeRate = 0/1024
        bitnum = 0
    elif SINR <= -5.1470 and SINR >= -6.9360:
        codeRate = 78/1024
        bitnum = 2
    elif SINR <= -3.1800 and SINR >= -5.1470:
        codeRate = 120/1024
        bitnum = 2
    elif SINR <= -1.2530 and SINR >= -3.1800:
        codeRate = 193/1024
        bitnum = 2
    elif SINR <= 0.7610 and SINR >= -1.2530:
        codeRate = 308/1024
        bitnum = 2
    elif SINR <= 2.6990 and SINR >= 0.7610:
        codeRate = 449/1024
        bitnum = 2
    elif SINR <= 4.6940 and SINR >= 2.6990:
        codeRate = 602/1024
        bitnum = 2
    elif SINR <= 6.5250 and SINR >= 4.6940:
        codeRate = 378/1024
        bitnum = 4
    elif SINR <= 8.5730 and SINR >= 6.5250:
        codeRate = 490/1024
        bitnum = 4
    elif SINR <= 10.3360 and SINR >= 8.5730:
        codeRate = 616/1024
        bitnum = 4
    elif SINR <= 12.2890 and SINR >= 10.3360:
        codeRate = 466/1024
        bitnum = 6
    elif SINR <= 14.1730 and SINR >= 12.2890:
        codeRate = 567/1024
        bitnum = 6
    elif SINR <= 15.8880 and SINR >= 14.1730:
        codeRate = 666/1024
        bitnum = 6
    elif SINR <= 17.8140 and SINR >= 15.8880:
        codeRate = 772/1024
        bitnum = 6
    elif SINR <= 19.8290 and SINR >= 17.8140:
        codeRate = 873/1024
        bitnum = 6
    else:
        codeRate = 948/1024
        bitnum = 6
    return codeRate,bitnum
@jit
def Link_rate(codeRate,bitnum):
    return math.ceil(codeRate*bitnum*144)
@jit
def D_RB(admittedDataRate,delayConstraint):
    return admittedDataRate*delayConstraint

