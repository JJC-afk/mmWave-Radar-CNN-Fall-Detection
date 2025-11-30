import struct
import serial
import os
import time, datetime
import pickle, json

from common_structs import InsufficientBytesError, CorruptMessage, CorruptTLV
from common_structs import  Session, Simple_struct, Multi_entry_struct, Multi_entry_preamble_struct

class Demo_TLV_header(Simple_struct):
    #For MmwDemo_output_message_tl
    #Contains 2 unsigned integers
    format_str = "II"
    attributes = ["type", "length"]


class Demo_TLV_body(Multi_entry_struct):
    pass
    

class Demo_pointcloud_cartesian(Demo_TLV_body):
    #For TLV type 1: MMWDEMO_OUTPUT_MSG_DETECTED_POINTS
    #For DPIF_PointCloudCartesian
    tlv_type = 1
    data_format_str = "ffff"
    data_attributes = ["x", "y", "z", "velocity"]


class Demo_range_profile(Demo_TLV_body):
    #For TLV type 2: MMWDEMO_OUTPUT_MSG_RANGE_PROFILE
    #For range profile
    tlv_type = 2
    data_format_str = "H"
    data_attributes = ["bin"]


class Demo_noise_profile(Demo_TLV_body):
    #For TLV type 3: MMWDEMO_OUTPUT_MSG_NOISE_PROFILE
    #For noise profile
    tlv_type = 3
    data_format_str = "H"
    data_attributes = ["bin"]


class Demo_azimuth_static_heatmap(Demo_TLV_body):
    #For TLV type 4: MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP
    #For azimuth static heatmap
    tlv_type = 4
    data_format_str = "hh"
    data_attributes = ["Im", "Re"]


class Demo_range_doppler_heatmap(Demo_TLV_body):
    #For TLV type 5: MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP
    #For range doppler heatmap
    tlv_type = 5
    data_format_str = "h"
    data_attributes = ["bin"]


class Demo_stats(Demo_TLV_body):
    #For TLV type 6: MMWDEMO_OUTPUT_MSG_STATS
    #For MmwDemo_output_message_stats
    tlv_type = 6
    data_format_str = "IIIIII"
    data_attributes = ["interFrameProcessingTime", "transmitOutputTime", "interFrameProcessingMargin",
                        "interChirpProcessingMargin", "activeFrameCPULoad", "interFrameCPULoad"]


class Demo_pointcloud_side_info(Demo_TLV_body):
    #For TLV type 7: MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO
    #For DPIF_PointCloudSideInfo
    tlv_type = 7
    data_format_str = "hh"
    data_attributes = ["snr", "noise"]


class Demo_azimuth_elevation_heatmap(Demo_TLV_body):
    #For TLV type 8: MMWDEMO_OUTPUT_MSG_AZIMUT_ELEVATION_STATIC_HEAT_MAP
    #For azimuth elevation heatmap
    tlv_type = 8
    data_format_str = "hh"
    data_attributes = ["Im", "Re"]


class Demo_temperature_stats(Demo_TLV_body):
    #For TLV type 9: MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS
    #For MmwDemo_temperatureStats
    tlv_type = 9
    data_format_str = "iIhhhhhhhhhh"
    data_attributes = ["tempReportValid", "time",
                        "tmpRx0Sens", "tmpRx1Sens", "tmpRx2Sens", "tmpRx3Sens",
                        "tmpTx0Sens", "tmpTx1Sens", "tmpTx2Sens",
                        "tmpPmSens", "tmpDig0Sens", "tmpDig1Sens"
                    ]


class Demo_TLV:

    #maps TLV type to which class represents the TLV body
    TLV_type_map = {tlv_body_class.tlv_type:tlv_body_class for
                        tlv_body_class in Demo_TLV_body.__subclasses__()}

    def __init__(self, byte_string, byte_ptr):

        self.header = Demo_TLV_header(byte_string, byte_ptr)
        byte_ptr += len(self.header)
        
        try:
            tlv_body_class = self.TLV_type_map[self.header.type]
        except KeyError:
            raise CorruptTLV

        if self.header.length % struct.calcsize(tlv_body_class.data_format_str) != 0:
            raise InsufficientBytesError

        data_count = self.header.length//struct.calcsize(tlv_body_class.data_format_str)
        self.body = tlv_body_class(byte_string, byte_ptr, data_count)
    
    def __len__(self):
        return len(self.header) + len(self.body)
    
    def get_dict(self):
        return {
                'header':self.header.get_dict(),
                'body' : self.body.get_dict()
                }


class Demo_message_header(Simple_struct):

    magic_word_correct = [0x0102, 0x0304, 0x0506, 0x0708]
    magic_word_str = b"\x02\x01\x04\x03\x06\x05\x08\x07"

    #For MmwDemo_output_message_header
    #Contains 4 unsigned shorts and 8 unsigned integers
    format_str = "HHHHIIIIIIII"
    attributes = [
                    "magic_word_0", "magic_word_1", "magic_word_2", "magic_word_3",
                    "version", "totalPacketLen", "platform", "frameNumber",
                    "timeCpuCycles", "numDetectedObj", "numTLVs", "subFrameNumber"
                ]
    
    def __init__(self, byte_string, byte_ptr):
        super().__init__(byte_string, byte_ptr)
        self.verify_magic_word()
    
    def verify_magic_word(self):
        #Raise corrupted message if magic word is incorrect
        if (
            self.magic_word_correct[0] != self.magic_word_0 or
            self.magic_word_correct[1] != self.magic_word_1 or
            self.magic_word_correct[2] != self.magic_word_2 or
            self.magic_word_correct[3] != self.magic_word_3
            ):
                raise CorruptMessage("Magic word did not match")


class Demo_message:

    def __init__(self, byte_string, byte_ptr):

        #Get header
        self.header = Demo_message_header(byte_string, byte_ptr)
        byte_ptr += len(self.header)

        #Put TLV array in body
        self.body = []
        for _ in range(self.header.numTLVs):
            try:
                self.body.append(Demo_TLV(byte_string, byte_ptr))
            except CorruptTLV:
                raise CorruptMessage(f"TLV type not recognized")
            byte_ptr += len(self.body[-1])
    
    def __len__(self):
        return self.header.totalPacketLen
    
    def get_dict(self):
        return {
                'header': self.header.get_dict(),
                'body' : [item.get_dict() for item in self.body]
                }


class Demo_session(Session):
    message_class = Demo_message

    def find_next_message(self, byte_string, byte_ptr):
        return byte_string.find(Demo_message_header.magic_word_str, byte_ptr+1)
