import mido
from mido import Message, MidiFile, MidiTrack

from src.process import check_msg_type, decoded_msg, MessageFormatter

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)


def get_message(array_as_input):
    xxx = {"note_status": array_as_input[0], "channel": array_as_input[1], "note": array_as_input[2],
           "velocity": array_as_input[3], "time": array_as_input[4], "unknown_message": array_as_input[5],
           "reset_message": array_as_input[6], "normal_message": array_as_input[7]}

    if xxx["reset_message"]: # reset message
        return Message("reset")
    elif xxx["normal_message"]: # normal message
        if xxx["note_status"]: # 1
            return Message('note_on', channel=xxx["channel"], note=xxx["note"], velocity=xxx["velocity"], time=xxx["time"])
        return Message('note_off', channel=xxx["channel"], note=xxx["note"], velocity=xxx["velocity"], time=xxx["time"])
    else: # unknown message
        # either drop it or convert them to reset messages
        return Message("reset") # converting to reset


def prepare_track(array_as_input):
    track.append(get_message(array_as_input))


def save_file():
    mid.save('new_file.mid')
