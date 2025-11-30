import struct
import serial
import os
import time, datetime
import pickle, json

class InsufficientBytesError(Exception):
    # Raised when byte_string, byte_ptr passed to __init__ do not represent sufficient bytes
    pass

class CorruptMessage(Exception):
    # Raised when the message being read is corrupt.
    pass

class CorruptTLV(Exception):
    # Raised when the TLV being read is corrupt.
    pass

class Session:
    # Class to store multiple messages in a recorded session.
    # Session needs to be extended. The extending class should have
    # a message_class attribute specifies the class whose objects are stored as messages
    # the message class should support byte string based initialization,
    # message class' __len__ function should return size of message object in bytes
    # message class should have a get_dict function
    # The message class' init function should raise InsufficientBytesError if it
    # the byte_string, byte_ptr combination provided does not have sufficient bytes for the message.

    def __init__(self, byte_string, byte_ptr):

        #Get all messages in recorded session
        self.messages = []
        while True:
            try:
                self.messages.append(self.message_class(byte_string, byte_ptr))
                byte_ptr += len(self.messages[-1])
            except InsufficientBytesError:
                #Last message might not be recorded completely
                break
            except CorruptMessage as e:
                print(e.__class__.__name__ +": "+ str(e))
                print(f"Saving message {len(self.messages)} as raw string")
                #Find next message to read
                next_byte_ptr = self.find_next_message(byte_string, byte_ptr)
                if next_byte_ptr == -1:
                    self.messages.append("RAW:"+str(byte_string[byte_ptr:]))
                    break
                else:
                    self.messages.append("RAW:"+str(byte_string[byte_ptr:next_byte_ptr]))
                byte_ptr = next_byte_ptr
    
    def __len__(self):
        return sum(map(len, self.messages))
    
    def get_dict(self):
        return {"messages" : [(item.get_dict() if hasattr(item, 'get_dict') else item) for item in self.messages]}


class Simple_struct:

    # Base class for simple struct comprising of basic datatypes handled by struct library
    # This class needs to be extended with the following class variables to work:
    #
    # format_str: A format string for the struct. Should be in accordance with the struct library
    # attributes: A list of attribute names for the fields. These names will become members of
    #             objects of the extended class.

    def __init__(self, byte_string, byte_ptr):
        try:
            tup = struct.unpack(self.format_str, 
                                byte_string[byte_ptr :
                                            byte_ptr+struct.calcsize(self.format_str)])
        except struct.error:
            raise InsufficientBytesError
        
        for i, attribute in enumerate(self.attributes):
            setattr(self, attribute, tup[i])
    
    def __len__(self):
        return struct.calcsize(self.format_str)

    def get_dict(self):
        return self.__dict__


class Multi_entry_struct:

    # Base class for struct comprising of basic datatypes handled by struct library
    # The struct has a data section and comprises of multiple data entries
    # A data entry comprises of a list of data attributes.
    #
    # This class needs to be extended with the following class variables to work:
    # get_data_count function: To return data count expected. Should exist if data_count is not given in __init__ function
    # data_format_str: A format string for a data entry in the data section of the struct. Should be in
    #                  accordance with the struct library
    # data_attributes: A list of attribute names for the data entry fields. These fields are keys for 
    #                  dicts. A dict represents an individual data entry. The data member variable of objects
    #                  of the extended class will contain an array of these dicts in the data member variable.

    def __init__(self, byte_string, byte_ptr, data_count=None):

        #Get data count
        self._data_count = data_count
        if data_count is None:
            self._data_count = self.get_data_count()

        #Format string for data section
        data_arr_format_str = self.data_format_str * self._data_count
        #Read data section as a tuple
        try:
            data_tup = struct.unpack(data_arr_format_str, 
                                    byte_string[byte_ptr : byte_ptr+
                                        struct.calcsize(data_arr_format_str)
                                        ]
                                    )
        except struct.error:
            raise InsufficientBytesError

        #Add data array of size self._data_count
        self.data = []
        tup_ptr=0
        for i in range(self._data_count):
            entry = dict()
            for attribute in self.data_attributes:
                entry[attribute] = data_tup[tup_ptr]
                tup_ptr += 1
            self.data.append(entry)

    def __len__(self):
        try:
            return struct.calcsize(self.data_format_str * self._data_count)
        except AttributeError:
            raise ValueError('object has not been filled with data')
    
    def get_dict(self):
        return {k:v for k,v in self.__dict__.items() if k[0] != '_'}


class Multi_entry_preamble_struct(Multi_entry_struct):
    # Base class for struct comprising of basic datatypes handled by struct library
    # The struct is divided into two sections: preamble and data. 
    # preamble starts at the beginning of the struct and contains multiple preamble attributes.
    # data comes after preamble. This section is part of the Multi_entry_struct class.
    #
    # This class needs to be extended with the following class variables to work:
    # preamble_format_str: A format string for the preamble section of the struct. Should be in accordance
    #                      with the struct library
    # preamble_attributes: A list of attribute names for the preamble fields. These names will
    #                      become members of objects of the extended class.
    # All attributes and functions required by Multi_entry_struct class.

    def __init__(self, byte_string, byte_ptr):

        #Read preamble section
        try:
            descr_tup = struct.unpack(
                            self.preamble_format_str, 
                            byte_string[byte_ptr : byte_ptr+
                                struct.calcsize(self.preamble_format_str)]
                            )
        except struct.error:
            raise InsufficientBytesError
        
        #Add preamble attributes
        for i, attribute in enumerate(self.preamble_attributes):
            setattr(self, attribute, descr_tup[i])
        
        #Advance binary string pointer to data section
        byte_ptr += struct.calcsize(self.preamble_format_str)

        #Read data section now
        super().__init__(byte_string, byte_ptr)
        

    def __len__(self):
        try:
            return struct.calcsize(self.preamble_format_str + self.data_format_str * self._data_count)
        except AttributeError:
            raise ValueError('object has not been filled with data')
    
    def get_dict(self):
        return self.__dict__


class Executor:

    def __init__(self, command_baud_rate, data_baud_rate,
                    command_port, data_port, commands,
                    session_class):
        self.command_baud_rate = command_baud_rate
        self.data_baud_rate = data_baud_rate
        self.command_port = command_port
        self.data_port = data_port
        self.commands = commands
        self.session_class = session_class

        self.raw_data = None
        self.parsed_data = None
        self.folder = None
    
    def send_commands(self):
        #Sends commands to start command port
        with serial.Serial(self.command_port, self.command_baud_rate, timeout=1) as ser:
            for command in self.commands:
                if command[0] == "%":
                    continue
                print(command)
                if command == "" or command[-1] != "\n":
                    ser.write((command+"\n").encode("utf-8"))
                else:
                    ser.write(command.encode("utf-8"))
                print(ser.read(1000).decode('utf-8'))
    
    def capture_data(self, duration, max_capture_size):
        #captures data for time seconds, upto max_size bytes
        with serial.Serial(self.data_port, self.data_baud_rate, timeout=duration) as ser:
            print(f'Data capture started on {self.data_port}')
            self.raw_data = ser.read(size=max_capture_size)
            
            return len(self.raw_data)
    
    def parse(self):
        session = self.session_class(self.raw_data, 0)
        self.parsed_data = session.get_dict()
    
    def set_dir(self, path, timestamp_subdir=True):
        self.folder = path
        if timestamp_subdir:
            #Make timestamped folder
            cur_time = time.time()
            cur_time = datetime.datetime.fromtimestamp(cur_time)
            cur_time = cur_time.strftime("%Y-%m-%d-%H-%M-%S")
            self.folder = os.path.join(self.folder, cur_time)
        
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
    
    def load_raw_data(self, path):
        with open(path, "rb") as f:
            self.raw_data = pickle.load(f)
    
    def save_raw_data(self):
        with open(os.path.join(self.folder, "raw.pickle"), "wb") as f:
            pickle.dump(self.raw_data, f)
    
    def save_parsed_data(self, index, indent=1):
        with open(os.path.join(self.folder, f"parsed ({index}).json"), "w") as f:
            if indent is not None:
                json.dump(self.parsed_data, f, indent=indent)
            else:
                json.dump(self.parsed_data, f)
