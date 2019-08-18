function   [labelp,fdata,delta,theta,alpha,beta] = read_one_file( filespec ) 
   
   p = h5read(filespec,'/labels/data');
   %labelp contain output label i.e.
   % 0->re , 1->ai or/a/ and 2->ui or /u/
   labelp = p(1);
   
   %data now contain the matrix [4 X 128] matrix that is 4 channels(filter)
   %data and each channel with 128 samples
   data = h5read(filespec, '/filteredEEG/data');
   
   %CSP Filter coefficients for the 4 channels
   %size of each data matrix is [1 X 128]
   data1 = data(1,:);
   data2 = data(2,:);
   data3 = data(3,:);
   data4 = data(4,:);
   
   %CSP filter coefficients combined to form a [1 X 512] feature matrix
   fdata = [data1 data2 data3 data4];
   
   % The frequency of the signal is 128 hz
   % Task is to find wavelet coefficients for all the bands of EEG
   % a.) Approximate coefficiet 5th level (0-4)hz is is DELTA BAND
   % b.) Detailed Coefficient of 5th level (4-8)hz is THETA BAND
   % c.) Detailed Coefficient of 4th level (8-16)hz is ALPHA BAND
   % d.) Detailed Coefficient of 3rd level (16-32)hz is BETA BAND
   
   
   % Currently using same wavelet family for all the bands
   wname = 'db2';
   
%1.)------------------------------------   
   % for DELTA BAND
   % a.) Approximate coefficiet 5th level (0-4)hz is is DELTA BAND
   Level = 5;
   [C,L] = wavedec(data1,Level,wname);
   app_coeff_ch1 = appcoef(C,L,wname,Level);
   [C,L] = wavedec(data2,Level,wname);
   app_coeff_ch2 = appcoef(C,L,wname,Level);
   [C,L] = wavedec(data3,Level,wname);
   app_coeff_ch3 = appcoef(C,L,wname,Level);
   [C,L] = wavedec(data4,Level,wname);
   app_coeff_ch4 = appcoef(C,L,wname,Level);
   
   % delta band coefficient
   delta = [app_coeff_ch1 app_coeff_ch2 app_coeff_ch3 app_coeff_ch4];
   
   
   
%2.)------------------------------------    
   % for THETA BAND
   % b.) Detailed Coefficient of 5th level (4-8)hz is THETA BAND
   Level = 5;
   [C1,L] = wavedec(data1,Level,wname);
   [C2,L] = wavedec(data2,Level,wname);
   [C3,L] = wavedec(data3,Level,wname);
   [C4,L] = wavedec(data4,Level,wname);
   
   % theta band coefficient
   theta = [C1 C2 C3 C4];
   
   
   
%3.)------------------------------------    
   % for ALPHA BAND
   % c.) Detailed Coefficient of 4th level (8-16)hz is ALPHA BAND
   Level = 4;
   [C1,L] = wavedec(data1,Level,wname);
   [C2,L] = wavedec(data2,Level,wname);
   [C3,L] = wavedec(data3,Level,wname);
   [C4,L] = wavedec(data4,Level,wname);
   
   % theta band coefficient
   alpha = [C1 C2 C3 C4];
   
   

%4.)------------------------------------    
   % for BETA BAND
   % d.) Detailed Coefficient of 3rd level (16-32)hz is BETA BAND
   Level = 3;
   [C1,L] = wavedec(data1,Level,wname);
   [C2,L] = wavedec(data2,Level,wname);
   [C3,L] = wavedec(data3,Level,wname);
   [C4,L] = wavedec(data4,Level,wname);
   
   % theta band coefficient
   beta = [C1 C2 C3 C4];
   
   
   
   
   
end
