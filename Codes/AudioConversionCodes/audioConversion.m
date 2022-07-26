
%function audioConversion(pathToParentToNumberedDir)
%.........................................................................%
% May'2017
% modified in May'2018
% Anindita Nath
% University of Texas at El Paso
%.........................................................................%
%converts .flac audios to .au format%
%.........................................................................%
%Input:
%   i)pathToParentToNumberedDir- absolute path to the parent of the numbered
%   directories. 
%   e.g. path to parent to '001',etc. directory 
%   
function audioConversion(pathToParentToNumberedDir)

    statusMsg=mkdir(pathToParentToNumberedDir,'ALL');%create a 'ALL' folder in the given path
    pathToALL=strcat(pathToParentToNumberedDir,'ALL\');
    statusMsg=mkdir(pathToALL,'converted'); % create a 'convered' folder under '..\ALL\'
    pathToALL=strrep(pathToALL,'\','/');
    
    audioDir=getfolders(pathToParentToNumberedDir);    

    for j=1:length(audioDir)
       
        pathToAudioFolder=strcat(pathToParentToNumberedDir,audioDir(j).name,'\','AUDIO\');       
        pathToFlacAudios=strcat(pathToAudioFolder,'*.au');
        flacFolders=dir(pathToFlacAudios);
        
         for k=1:length(flacFolders)       

                audioName=flacFolders(k).name;
                audioPath=strcat(pathToAudioFolder,audioName);

                audioPath = strrep(audioPath,'\','/');  
                audioPathstr=cellstr(audioPath);%string format of the path.... 
                                                %to the audio file to be converted
                pathToALL=strrep(pathToALL,'\','/');
                pathToALLstr=cellstr(pathToALL);% string format of the path....
                                                %to 'ALL' where files to be converetd are placed
                %move the .flac audios from each numbered directories to
                %'..ALL\'
                movefile (audioPathstr{1,1},pathToALLstr{1,1},'f');% 'source-string', ....
                                                                   %'destinaion-string',....
                                                                   %'f' for write permission               
                
           
% doesnot work form within this script, have to run manually         
%              command='convertAU.bat *.flac';
%              status = system(command);
%                  system('!convertAU.bat *.flac');       
  
         end       
    end  
end



