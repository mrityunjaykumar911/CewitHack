import mido
from mido.messages import messages
import json


def check_if_file_exists(function_name):
    def check_it(*args, **kwargs):
        ip_file = args[0]
        import os
        assert os.path.exists(ip_file)
        ret = function_name(*args, **kwargs)
        return ret

    return check_it


@check_if_file_exists
def open_midi_file(midi_file_path):
    return mido.MidiFile(midi_file_path)


class BaseTensor:
    def __init__(self):
        self.signature = "BaseTensor"
        self.arr = {"note_status": 0,
                    "channel": 0,
                    "note": 0,
                    "velocity": 0,
                    "time": 0,
                    "type": "DummyType"}

    def __setitem__(self, key, value):
        if key in self.arr.keys():
            self.arr[key] = value

    def get_arr(self):
        return [self.arr["note_status"],
                self.arr["channel"],
                self.arr["note"],
                self.arr["velocity"],
                self.arr["time"],
                self.arr["type"]]

    def __repr__(self):
        return str(self.get_arr())

    def __str__(self):
        return json.dumps(self.arr)


class UnknownMsgType(BaseTensor):
    def __init__(self):
        super().__init__()
        self.signature = "UNK"


class ResetTimerMsg(BaseTensor):
    """
    message reset time=0
    """

    def __init__(self, string_as_inp):
        super().__init__()
        string_as_inp = string_as_inp.replace("[", "")
        string_as_inp = string_as_inp.replace("<", "")
        string_as_inp = string_as_inp.replace(">", "")
        string_as_inp = string_as_inp.replace("]", "")

        self.signature = "reset"
        get_only_time = int(string_as_inp[string_as_inp.rindex("=") + 1:])
        self.arr["type"] = "ResetMsg"
        self.arr["time"] = get_only_time


class MessageFormatter(BaseTensor):
    """
    message note_on channel=9 note=38 velocity=0 time=0

    """

    def __init__(self, string_as_ip):
        super().__init__()
        self.signature = "NormalMsg"
        # self.tensor = BaseTensor()
        trimmed = string_as_ip.replace("[", "")
        trimmed = trimmed.replace("]", "")
        trimmed = trimmed.replace("<", "")
        trimmed = trimmed.replace(">", "")

        len_must_be_six = len(trimmed.split(" "))
        assert len_must_be_six == 6
        first, second, third, four, five, six = trimmed.split(" ")
        assert first == "message"

        self.arr["type"] = "NormalMsg"
        self.arr["channel"] = int(third[third.rindex("=") + 1:])
        self.arr["note_status"] = 1 if second == "note_on" else 0
        self.arr["note"] = int(four[four.rindex("=") + 1:])
        self.arr["velocity"] = int(five[five.rindex("=") + 1:])
        self.arr["time"] = int(six[six.rindex("=") + 1:])


global_msg_count = 0
mid3 = open_midi_file("../data/actual_midi/twokb.mid")


def check_msg_type(msg):
    if "reset time" in msg:
        return ResetTimerMsg
    elif "velocity" in msg and "channel" in msg:
        return MessageFormatter
    else:
        return UnknownMsgType


all_data_points = {}
for i, track in enumerate(mid3.tracks):
    print("--*--" * 20)
    print('Track {}: {}'.format(i, track.name))
    # print(track,dir(track))
    len_msg = len(track)

    for msg in track:
        global_msg_count += 1
        bytes_p = msg.bytes()

        decoded_msg = str(mido.parse_all(bytes_p))
        print(decoded_msg)

        msg_type = check_msg_type(msg=decoded_msg)
        if msg_type == MessageFormatter:
            identifier = "M"
            msg_repr = MessageFormatter(decoded_msg)
        elif msg_type == ResetTimerMsg:
            identifier = "R"
            msg_repr = ResetTimerMsg(decoded_msg)
        else:
            identifier = "U"
            msg_repr = UnknownMsgType()

        all_data_points["%s_%s"%(identifier,global_msg_count)] = msg_repr.arr

        # print("messages length={} track_id={}".format(len_msg, i + 1))
    print("global messages count={}".format(global_msg_count))

# save as json
# fp = open("data_v1.json","w")
# json.dump(all_data_points,fp)
# fp.close()
