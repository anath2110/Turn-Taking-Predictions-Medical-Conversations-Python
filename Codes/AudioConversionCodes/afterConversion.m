%function afterConversion(pathToParentToNumberedDir)
%.........................................................................%
% May'2017
% modified in May'2018
% Anindita Nath
% University of Texas at El Paso
%.........................................................................%
%puts back the converted .au files to the corresponding numbered
%directories
%.........................................................................%
%Input:
%   i)pathToParentToNumberedDir- absolute path to the parent of the numbered
%   directories. 
%   e.g. path to parent to '001',etc. directory 
%   
function afterConversion(pathToParentToNumberedDir)
pathToAll=strcat(pathToParentToNumberedDir,'ALL');
pathToConvertedAudios=strcat(pathToParentToNumberedDir,'ALL\converted\');
%pathToConvertedAudios=strcat(pathToParentToNumberedDir,'pitchCache\');
audiofolders=getfolders(pathToParentToNumberedDir);
countofaudio=length(audiofolders);

for k=1:countofaudio

   auDirName=audiofolders(k).name;
   pathToAudioFolder=strcat(pathToParentToNumberedDir,auDirName,'\','AUDIO\');
   pathToAudioFolder=strrep(pathToAudioFolder,'\','/');
   pathToAudioFolderstr=cellstr(pathToAudioFolder);
   %statusMsg=mkdir(pathToAudioFolder,'pitchCache');
 %after conversion : copy each converted file from '..\ALL\converted\' to respective directories
  audiosConverted=dir(strcat(pathToConvertedAudios,'*.au'));
  %audiosConverted=dir(strcat(pathToConvertedAudios,'*.mat'));
       for z=1:length(audiosConverted)
           
            last3OfAudioName=audiosConverted(z).name(end-9:end-7); %'001','002', etc.
            %last3OfAudioName=audiosConverted(z).name(end-10:end-8); %'001','002', etc.
            %disp(last3OfAudioName);
            auName=audiosConverted(z).name;%ARA_001%
            auNamePath=strcat(pathToConvertedAudios,auName);
            auNamePathstr=cellstr(auNamePath);          
            
             if (strcmp(auDirName,last3OfAudioName)>0)
             %move the converted .au audios from '..ALL\converted\' to corresponding numbered directory
                 
                 %destination=strcat(pathToAudioFolderstr{1,1},'/','pitchCache');
                 %disp(destination);
                 movefile(auNamePathstr{1,1},pathToAudioFolderstr{1,1});
                 %movefile(auNamePathstr{1,1},destination);
                 disp('Success');
                 
             else
                 %pass
             end
                 
       end
end
%status = rmdir(pathToAll);   % remove '..\ALL\converted'.  
end

function folders = getfolders(path)
%get all the directories in the given path except..
%current directory '.' and %parent directory '..'
folders = dir(path);

for k = length(folders):-1:1
    % remove non-folders
    if ~folders(k).isdir
        folders(k) = [ ];
        continue
    end

    % remove folders starting with .
    fname = folders(k).name;
    if fname(1) == '.'
        folders(k) = [ ];
    end
end

end

