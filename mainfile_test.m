clear;close all;
clc;

data = data_from_multiple_HDF5_files( 'C:\Users\ece\Desktop\Rajiv\SupplementaryFiles', '*.h5' );

disp('-------------------------------------------------------------------');
% Structure for storing performance metric
performance_struct = struct('subject',[],'task',[],'classifier',[],'accuracy',[],'std',[],'precision',[],'recall',[],'f1score',[]);

% Convert the structure into table
% Accessing element from table is similar to array so it eases manipulation
data = struct2table(data);

% CSP COEFFICIENTS 
csp_coeff = data.fdata;

% features extracted directly from csp coefficients 
energy_csp = [sum(csp_coeff(:,1:128).^2,2) sum(csp_coeff(:,129:256).^2,2)  sum(csp_coeff(:,257:384).^2,2) sum(csp_coeff(:,385:512).^2,2)];
rms_csp = [sqrt(sum(csp_coeff(:,1:128).^2,2)./128) sqrt(sum(csp_coeff(:,129:256).^2,2)./128) sqrt(sum(csp_coeff(:,257:384).^2,2)./128) sqrt(sum(csp_coeff(:,385:512).^2,2)./128)];
sum_csp = [sum(csp_coeff(:,1:128),2) sum(csp_coeff(:,129:256),2) sum(csp_coeff(:,257:384),2) sum(csp_coeff(:,385:512),2)];
std_csp = [std(csp_coeff(:,1:128)')' std(csp_coeff(:,129:256)')' std(csp_coeff(:,257:384)')' std(csp_coeff(:,385:512)')'];
%mean_csp = [mean(csp_coeff(:,1:128),2) mean(csp_coeff(:,129:256),2) mean(csp_coeff(:,257:384),2) mean(csp_coeff(:,385:512),2)];


% response variable output
y = data.label;

%beta coefficients
% x1 = data.beta;
% x2 = data.beta;
% x3 = data.beta;
% x4 = data.beta;
% X_beta =[x1,x2,x3,x4];

% features extracted from beta wavelet coefficients
% rms_beta = [sqrt(sum(x1.^2,2)./size(x1,2)) sqrt(sum(x2.^2,2)./size(x2,2)) sqrt(sum(x3.^2,2)./size(x3,2)) sqrt(sum(x4.^2,2)./size(x4,2))];
% sum_beta = [sum(x1,2) sum(x2,2) sum(x3,2) sum(x4,2)];
% energy_beta = [sum(x1.^2,2) sum(x2.^2,2) sum(x3.^2,2) sum(x4.^2,2)];
% std_beta = [std(x1')' std(x2')' std(x3')' std(x4')'];
% beta_features = [rms_beta sum_beta energy_beta std_beta];



%alpha coefficients
% x1 = data.alpha;
% x2 = data.alpha;
% x3 = data.alpha;
% x4 = data.alpha;
% X_alpha =[x1,x2,x3,x4];

% rms_alpha = [sqrt(sum(x1.^2,2)./size(x1,2)) sqrt(sum(x2.^2,2)./size(x2,2)) sqrt(sum(x3.^2,2)./size(x3,2)) sqrt(sum(x4.^2,2)./size(x4,2))];
% sum_alpha = [sum(x1,2) sum(x2,2) sum(x3,2) sum(x4,2)];
% energy_alpha = [sum(x1.^2,2) sum(x2.^2,2) sum(x3.^2,2) sum(x4.^2,2)];
% std_alpha = [std(x1')' std(x2')' std(x3')' std(x4')'];
% 

%theta coefficients
% x1 = data.theta;
% x2 = data.theta;
% x3 = data.theta;
% x4 = data.theta;
% 
% rms_theta = [sqrt(sum(x1.^2,2)./size(x1,2)) sqrt(sum(x2.^2,2)./size(x2,2)) sqrt(sum(x3.^2,2)./size(x3,2)) sqrt(sum(x4.^2,2)./size(x4,2))];
% sum_theta = [sum(x1,2) sum(x2,2) sum(x3,2) sum(x4,2)];
% energy_theta = [sum(x1.^2,2) sum(x2.^2,2) sum(x3.^2,2) sum(x4.^2,2)];
% std_theta = [std(x1')' std(x2')' std(x3')' std(x4')'];
% X_theta =[x1,x2,x3,x4];

%delta coefficients
% x1 = data.delta;
% x2 = data.delta;
% x3 = data.delta;
% x4 = data.delta;
% 
% rms_delta = [sqrt(sum(x1.^2,2)./size(x1,2)) sqrt(sum(x2.^2,2)./size(x2,2)) sqrt(sum(x3.^2,2)./size(x3,2)) sqrt(sum(x4.^2,2)./size(x4,2))];
% sum_delta = [sum(x1,2) sum(x2,2) sum(x3,2) sum(x4,2)];
% energy_delta = [sum(x1.^2,2) sum(x2.^2,2) sum(x3.^2,2) sum(x4.^2,2)];
% std_delta = [std(x1')' std(x2')' std(x3')' std(x4')'];
% X_delta =[x1,x2,x3,x4];


% we are going to apply 3 models for all the four bands
% features are wavelet coefficient of  4 bands taken one at a time
%---------MODELS----------
% 1.) Linear SVM
% 2.) Medium Gaussian SVM
% 3.) Ensemble Bagged tree

% we are using 20 fold cross validation
k_fold = 5;

% % %============================= S1==========================================
% % %---------------------- SUBJECT 1 DATA ANALYSIS ---------------------------
% % %==========================================================================
% % ------------------ S1 aire training and test -----------------------------
% performace_b11 = KFoldCrossValidaion(k_fold, X_beta(1:96,:) ,y(1:96));
% performace_a11 = KFoldCrossValidaion(k_fold, X_alpha(1:96,:) ,y(1:96));
% performace_t11 = KFoldCrossValidaion(k_fold, X_theta(1:96,:) ,y(1:96));
% performace_d11 = KFoldCrossValidaion(k_fold, X_delta(1:96,:) ,y(1:96));

% testing on the 11 set
csp_feature_11 =  [energy_csp rms_csp sum_csp std_csp];
%abtd_feature_11 = [sum_beta, energy_beta, rms_beta, std_beta, sum_alpha, energy_alpha, rms_alpha, std_alpha, sum_delta, energy_delta , rms_delta, std_delta, sum_theta, energy_theta, rms_theta, std_theta ];
performance_11 = KFoldCrossValidaion(k_fold, csp_feature_11(1:96,:)  ,y(1:96));
%performace_b_11 = KFoldCrossValidaion(k_fold, beta_features(1:96,:)  ,y(1:96));


% 
% % % ------------------S1 aiui training and test -----------------------------
% performace_b12 = KFoldCrossValidaion(k_fold, X_beta(97:196,:) ,y(97:196));
% performace_a12 = KFoldCrossValidaion(k_fold, X_alpha(97:196,:) ,y(97:196));
% performace_t12 = KFoldCrossValidaion(k_fold, X_theta(97:196,:) ,y(97:196));
% performace_d12 = KFoldCrossValidaion(k_fold, X_delta(97:196,:) ,y(97:196));
performance_12 = KFoldCrossValidaion(k_fold, csp_feature_11(97:196,:)  ,y(97:196));
% 
% % % ------------------S1 uire training and test -----------------------------
% performace_b13 = KFoldCrossValidaion(k_fold, X_beta(197:294,:) ,y(197:294));
% performace_a13 = KFoldCrossValidaion(k_fold, X_alpha(197:294,:) ,y(197:294));
% performace_t13 = KFoldCrossValidaion(k_fold, X_theta(197:294,:) ,y(197:294));
% performace_d13 = KFoldCrossValidaion(k_fold, X_delta(197:294,:) ,y(197:294));
performance_13 = KFoldCrossValidaion(k_fold, csp_feature_11(197:294,:)  ,y(197:294));
% 
% 
% 
% % %============================= S2==========================================
% % %---------------------- SUBJECT 2 DATA ANALYSIS ---------------------------
% % %==========================================================================
% % % ------------------ S2 aire training and test ----------------------------
% performace_b21 = KFoldCrossValidaion(k_fold, X_beta(295:394,:) ,y(295:394));
% performace_a21 = KFoldCrossValidaion(k_fold, X_alpha(295:394,:) ,y(295:394));
% performace_t21 = KFoldCrossValidaion(k_fold, X_theta(295:394,:) ,y(295:394));
% performace_d21 = KFoldCrossValidaion(k_fold, X_delta(295:394,:) ,y(295:394));
performance_21 = KFoldCrossValidaion(k_fold, csp_feature_11(295:394,:)  ,y(295:394));
% 
% 
% % % ------------------S2 aiui training and test -----------------------------
% performace_b22 = KFoldCrossValidaion(k_fold, X_beta(395:434,:) ,y(395:434));
% performace_a22 = KFoldCrossValidaion(k_fold, X_alpha(395:434,:) ,y(395:434));
% performace_t22 = KFoldCrossValidaion(k_fold, X_theta(395:434,:) ,y(395:434));
% performace_d22 = KFoldCrossValidaion(k_fold, X_delta(395:434,:) ,y(395:434));
performance_22 = KFoldCrossValidaion(k_fold, csp_feature_11(395:494,:)  ,y(395:494));
%  
% % % ------------------S2 uire training and test -----------------------------
% performace_b23 = KFoldCrossValidaion(k_fold, X_beta(495:534,:) ,y(495:534));
% performace_a23 = KFoldCrossValidaion(k_fold, X_alpha(495:534,:) ,y(495:534));
% performace_t23 = KFoldCrossValidaion(k_fold, X_theta(495:534,:) ,y(495:534));
% performace_d23 = KFoldCrossValidaion(k_fold, X_delta(495:534,:) ,y(495:534));
performance_23 = KFoldCrossValidaion(k_fold, csp_feature_11(495:594,:)  ,y(495:594));
% 
% 
% 
% % %============================= S3==========================================
% % %---------------------- SUBJECT 3 DATA ANALYSIS ---------------------------
% % %==========================================================================
% performace_b31 = KFoldCrossValidaion(k_fold, X_beta(595:693,:) ,y(595:693));
% performace_a31 = KFoldCrossValidaion(k_fold, X_alpha(595:693,:) ,y(595:693));
% performace_t31 = KFoldCrossValidaion(k_fold, X_theta(595:693,:) ,y(595:693));
% performace_d31 = KFoldCrossValidaion(k_fold, X_delta(595:693,:) ,y(595:693));
performance_31 = KFoldCrossValidaion(k_fold, csp_feature_11(595:693,:)  ,y(595:693));
% 
% 
% % % ------------------S3 aiui training and test -----------------------------
% performace_b32 = KFoldCrossValidaion(k_fold, X_beta(694:793,:) ,y(694:793));
% performace_a32 = KFoldCrossValidaion(k_fold, X_alpha(694:793,:) ,y(694:793));
% performace_t32 = KFoldCrossValidaion(k_fold, X_theta(694:793,:) ,y(694:793));
% performace_d32 = KFoldCrossValidaion(k_fold, X_delta(694:793,:) ,y(694:793));
performance_32 = KFoldCrossValidaion(k_fold, csp_feature_11(694:793,:)  ,y(694:793));
% 
% 
% % % ------------------S3 uire training and test -----------------------------
% performace_b33 = KFoldCrossValidaion(k_fold, X_beta(794:893,:) ,y(794:893));
% performace_a33 = KFoldCrossValidaion(k_fold, X_alpha(794:893,:) ,y(794:893));
% performace_t33 = KFoldCrossValidaion(k_fold, X_theta(794:893,:) ,y(794:893));
% performace_d33 = KFoldCrossValidaion(k_fold, X_delta(794:893,:) ,y(794:893));
performance_33 = KFoldCrossValidaion(k_fold, csp_feature_11(794:893,:)  ,y(794:893));
% 



% Converting structure performance to format suitable for converting to 
% csv file
for i=1:3
    performance_struct(i).subject = 'S1';
    performance_struct(i).task = 'aire';
    performance_struct(i).classifier = performance_11(i).classifier;
    performance_struct(i).accuracy = performance_11(i).accuracy;
    performance_struct(i).precision = performance_11(i).precision;
    performance_struct(i).recall = performance_11(i).recall;
    performance_struct(i).std = performance_11(i).std;
    performance_struct(i).f1score = performance_11(i).f1Score;
end


for i=4:6
    performance_struct(i).subject = 'S1';
    performance_struct(i).task = 'aiui';
    performance_struct(i).classifier = performance_12(i-3).classifier;
    performance_struct(i).accuracy = performance_12(i-3).accuracy;
    performance_struct(i).precision = performance_12(i-3).precision;
    performance_struct(i).recall = performance_12(i-3).recall;
    performance_struct(i).std = performance_12(i-3).std;
    performance_struct(i).f1score = performance_12(i-3).f1Score;
end


for i=7:9
    performance_struct(i).subject = 'S1';
    performance_struct(i).task = 'uire';
    performance_struct(i).classifier = performance_13(i-6).classifier;
    performance_struct(i).accuracy = performance_13(i-6).accuracy;
    performance_struct(i).precision = performance_13(i-6).precision;
    performance_struct(i).recall = performance_13(i-6).recall;
    performance_struct(i).std = performance_13(i-6).std;
    performance_struct(i).f1score = performance_13(i-6).f1Score;
end


for i=10:12
    performance_struct(i).subject = 'S2';
    performance_struct(i).task = 'aire';
    performance_struct(i).classifier = performance_21(i-9).classifier;
    performance_struct(i).accuracy = performance_21(i-9).accuracy;
    performance_struct(i).precision = performance_21(i-9).precision;
    performance_struct(i).recall = performance_21(i-9).recall;
    performance_struct(i).std = performance_21(i-9).std;
    performance_struct(i).f1score = performance_21(i-9).f1Score;
end


for i=13:15
    performance_struct(i).subject = 'S2';
    performance_struct(i).task = 'aiui';
    performance_struct(i).classifier = performance_22(i-12).classifier;
    performance_struct(i).accuracy = performance_22(i-12).accuracy;
    performance_struct(i).precision = performance_22(i-12).precision;
    performance_struct(i).recall = performance_22(i-12).recall;
    performance_struct(i).std = performance_22(i-12).std;
    performance_struct(i).f1score = performance_22(i-12).f1Score;
end


for i=16:18
    performance_struct(i).subject = 'S2';
    performance_struct(i).task = 'uire';
    performance_struct(i).classifier = performance_23(i-15).classifier;
    performance_struct(i).accuracy = performance_23(i-15).accuracy;
    performance_struct(i).precision = performance_23(i-15).precision;
    performance_struct(i).recall = performance_23(i-15).recall;
    performance_struct(i).std = performance_23(i-15).std;
    performance_struct(i).f1score = performance_23(i-15).f1Score;
end


for i=19:21
    performance_struct(i).subject = 'S3';
    performance_struct(i).task = 'aire';
    performance_struct(i).classifier = performance_31(i-18).classifier;
    performance_struct(i).accuracy = performance_31(i-18).accuracy;
    performance_struct(i).precision = performance_31(i-18).precision;
    performance_struct(i).recall = performance_31(i-18).recall;
    performance_struct(i).std = performance_31(i-18).std;
    performance_struct(i).f1score = performance_31(i-18).f1Score;
end


for i=22:24
    performance_struct(i).subject = 'S3';
    performance_struct(i).task = 'aiui';
    performance_struct(i).classifier = performance_32(i-21).classifier;
    performance_struct(i).accuracy = performance_32(i-21).accuracy;
    performance_struct(i).precision = performance_32(i-21).precision;
    performance_struct(i).recall = performance_32(i-21).recall;
    performance_struct(i).std = performance_32(i-21).std;
    performance_struct(i).f1score = performance_32(i-21).f1Score;
end


for i=25:27
    performance_struct(i).subject = 'S3';
    performance_struct(i).task = 'uire';
    performance_struct(i).classifier = performance_33(i-24).classifier;
    performance_struct(i).accuracy = performance_33(i-24).accuracy;
    performance_struct(i).precision = performance_33(i-24).precision;
    performance_struct(i).recall = performance_33(i-24).recall;
    performance_struct(i).std = performance_33(i-24).std;
    performance_struct(i).f1score = performance_33(i-24).f1Score;
end

% converting struct to table which then can be converted to csv file
output = struct2table(performance_struct);
writetable(output,'output_12_06_2017_11_51.csv');

disp('-----------FILE RUNNING COMPLETE-----------------');