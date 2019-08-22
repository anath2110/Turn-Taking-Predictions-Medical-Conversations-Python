# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:02:57 2019

@author:Anindita Nath 
        3M|M*Modal
        Pittsburgh,PA        
"""
"""
Function to parse a stm file.
Input: stm file
Output: list of all segemnts, doctor segments and patient segments
"""
import codecs
import unicodedata

class STMSegment(object):
    """
    Representation of an individual segment in an STM file.
    """
    def __init__(self, stm_line):
        tokens = stm_line.split()
        self._filename    = tokens[0]
        self._speakerType = tokens[1]
        self._audioNo     = tokens[2]
        self._start_time  = float(tokens[3])
        self._stop_time   = float(tokens[4])       
        self._transcript  = ""
        for token in tokens[5:]:
          self._transcript += token + " "
        # We need to do the encode-decode dance here because encode
        # returns a bytes() object on Python 3, and text_to_char_array
        # expects a string.
        self._transcript = unicodedata.normalize("NFKD", self._transcript.strip())  \
                                      .encode("ascii", "ignore")                    \
                                      .decode("ascii", "ignore")

    @property
    def filename(self):
        return self._filename

    @property
    def speakerType(self):
        return self._speakerType

    @property
    def audioNo(self):
        return self._audioNo

    @property
    def start_time(self):
        return self._start_time

    @property
    def stop_time(self):
        return self._stop_time
   

    @property
    def transcript(self):
        return self._transcript

def parse_stm_file(stm_file):
    """
    Parses an STM file at 'stm_file' into a list of objects of class:'STMSegment'.
    """
    global dialogflag
    dialogflag=1
    stm_segments = []
    doctor_segments  = []
    patient_segments = []
#    print("Start of Program")
#    print(stm_file)
    try:
        with codecs.open(stm_file, encoding="utf-8") as stm_lines:
            for stm_line in stm_lines:
                stmSegment = STMSegment(stm_line)
                #print(stmSegment.stop_time)
                if not "ignore_time_segment_in_scoring" == stmSegment.transcript:
                    stm_segments.append(stmSegment)
                    if  "DR" == stmSegment.speakerType:
                        doctor_segments.append(stmSegment)
                    elif("PT" == stmSegment.speakerType):
                        patient_segments.append(stmSegment)
                    elif(stmSegment.speakerType!="DR" or stmSegment.speakerType!="PT"):
                        dialogflag=2
#                    print("There's more than 1 speaker, not a dialog!")
#                    print(stmSegment.filename)
    except OSError:        
        pass
        
                    
    if(len(patient_segments)==0 or len(doctor_segments)==0):
        dialogflag=0
#        print("There's only 1 speaker, not a dialog!")
#        print(stmSegment.filename)
    
     #only if it is a dialog, i.e. has both doctors and patient segemnts
     #and only these 2 segments, change the dialog flag to 1
     # and print the name of the stm file to std out 
     #and then redirect stdout to a text file, 
     #to generate a list of dialog stm files
     
    if(dialogflag==1):
        print(stm_file)      
    return stm_segments,doctor_segments,patient_segments

#stm_segments,doctors_segemnts,patients_segments=parse_stm_file("D:/FUNDataCodes/NewTurnStm/oneStm/118028.stm")
#print(len(doctors_segemnts))
#print(doctors_segemnts[107].speakerType)
#print(doctors_segemnts[107].start_time)
#print(doctors_segemnts[107].stop_time)
#print(doctors_segemnts[107].transcript)

#print(len(stm_segments))
#print(stm_segments[10].speakerType)
#print(stm_segments[10].start_time)
#print(stm_segments[10].stop_time)
#print(stm_segments[10].transcript)