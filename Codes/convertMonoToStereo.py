# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:14:39 2019
@author: Anindita Nath
3M| M*Modal
Pittsburgh, PA
"""
"""
Converts only wav dialogs but with 1 channel to 2 channel stereos
Duplicates single channel data to both channels
"""
import wave, array
def make_stereo(inaudio, outaudio):
    ifile = wave.open(inaudio)
    #print (ifile.getparams())
    
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = ifile.getparams()
    assert comptype == 'NONE'  # Compressed not supported yet
    array_type = {1:'B', 2: 'h', 4: 'l'}[sampwidth]
    left_channel = array.array(array_type, ifile.readframes(nframes))[::nchannels]
    ifile.close()

    stereo = 2 * left_channel
    stereo[0::2] = stereo[1::2] = left_channel

    ofile = wave.open(outaudio, 'w')
    ofile.setparams((2, sampwidth, framerate, nframes, comptype, compname))
    #ofile.writeframes(stereo.tostring()) #depreciated, hence tobyte() used
    ofile.writeframes(stereo.tobytes())
    ofile.close()
#make_stereo("D:/FUNDataCodes/Audios/100002.wav", "D:/FUNDataCodes/Audios/ProcessedByPython/100002.wav")

