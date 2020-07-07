def raw2temp():
 try:
     import os # Package for working with OS
     import numpy as np # Package for working with matrices
     import csv
     import sys # System package
     import math
     import tifffile # Export to TIFF
     print('Import successful')
 except:
    print('Import failed')
    
 # FOLDER_RAW: source folder including CSV with RAW data
 # RAW_VIGNETTE: folder where mask is going to be stored
 
 FOLDER_RAW = r''
 FOLDER_VIGNETTE = r''
 
 #final_raw: final matrix
 #raw_matrix: matrix with loaded and opened CSV files
 #mask_matrix: matrix where vignetting mask is loaded
 final_raw = np.array([])
 raw_matrix = np.array([])
 mask_matrix = np.array([])
 
 E = 0.98 # emisivity
 OD = 70.0 # object distance
 RTemp = 15.0 # reflected temperature
 ATemp = RTemp # temperature of atmosphere
 IRWTemp = RTemp # temperature of optics
 IRT = 1.0 # filter
 RH = 94.0 # relative humidity
 
 ATA1 = 0.006569 # atmosherical constant by camera
 ATA2 = 0.012620
 ATB1 = -0.002276
 ATB2 = -0.006670
 ATX = 1.900000
 
 PR1 = 16556 # Plank´s constants for calibration by factory
 PB = 1428
 PF = 1
 PO = -342
 PR2 = 0.045167
 
 emisivity_wind = 1.0 - IRT
 reflectance_wind = 0
 
 # atmospehrical constants
 h2o_vapour = (RH / 100) * math.exp(1.5587 + 0.06939 * (ATemp) - 0.00027816 * (ATemp)** 2 + 0.00000068455 * (ATemp) ** 3)
 tau1 = ATX * math.exp((math.sqrt(OD / 2) * -1) * (ATA1 + ATB1 *math.sqrt(h2o_vapour))) + (1 - ATX) * math.exp((math.sqrt(OD / 2) * -1) * (ATA2 + ATB2 * math.sqrt(h2o_vapour)))
 tau2 = ATX * math.exp((math.sqrt(OD / 2) * -1) * (ATA1 + ATB1 *math.sqrt(h2o_vapour))) + (1 - ATX) * math.exp((math.sqrt(OD / 2) * -1) * (ATA2 + ATB2 * math.sqrt(h2o_vapour)))
 
 files = []
 mask = []
 raw_list = []
 list_of_temp = []
 
 for file in os.listdir(FOLDER_VIGNETTE):
    if file.endswith('.csv'):
        files.append(file)
        
 calibration_files = []
 
 for csv_file in files:
    calibration_files.append(np.genfromtxt(open(os.path.join(FOLDER_VIGNETTE, csv_file), "rb"),delimiter=";", dtype=np.uint16))
    calibration_files = np.array(calibration_files)
 
 # Matrix consisting of mean of each cell from calibration files
 mean_matrix = np.array(np.mean(calibration_files, axis=0))
 max_value = np.max(mean_matrix)
 
 # Substraction of mean and maximum values
 mask = np.rint(np.subtract(mean_matrix, max_value)) * (-1)
 mask_csv = np.savetxt('mask.csv', mask, delimiter=';')
 
 # Loading created mask
 for csv_file2 in os.listdir(FOLDER_RAW):
    if csv_file2.endswith('.csv'):
        raw_list.append(csv_file2)
 mask_matrix = np.loadtxt('mask.csv', delimiter=';', dtype=np.uint32)
 
 # Open each CSV with RAW DATA as string, replacing the decimals and substracting the last empty column
 for csv_file2 in raw_list:
    raw_matrix = np.genfromtxt(open(os.path.join(FOLDER_RAW, csv_file2)), delimiter=';', dtype=str)
    raw_matrix = np.char.replace(raw_matrix, ',000', '')
    raw_matrix = raw_matrix[:, :-1]
    raw_matrix = raw_matrix.astype(np.uint32)
    final_raw = np.add(mask_matrix, raw_matrix)
    
 # radiance from the environment
 # Originally made by h3ct0r, available at:https://github.com/Nervengift/read_thermal.py/blob/master/flir_image_extractor.py
 
 raw_refll = PR1 / (PR2 * (math.exp(PB / (RTemp + 273.15) - PF)) - PO)
 raw_refl1_attn = (1.0 - E) / (E * raw_refll)
 raw_atm1 = PR1 / (PR2 * (math.exp(PB / (ATemp + 273.15)) - PF)) - PO
 raw_atm1_attn = ((1.0 - tau1) / E) / (tau1 * raw_atm1)
 raw_wind = PR1 / (PR2 * (math.exp(PB / (IRWTemp + 273.15)) - PF)) - PO
 raw_wind_attn = (emisivity_wind / E) / (tau1 / IRT * raw_wind)
 raw_refl2 = PR1 / (PR2 * (math.exp(PB / (RTemp + 273.15)) - PF)) - PO
 raw_refl2_attn = reflectance_wind / E / tau1 / IRT * raw_refl2
 raw_atm2 = PR1 / (PR2 * (math.exp(PB / (ATemp + 273.15)) - PF)) - PO
 raw_atm2_attn = (1.0 - tau2) / E / tau1 / IRT / tau2 * raw_atm2
 raw_obj = (final_raw / E / tau1 / IRT / tau2 - raw_atm1_attn - raw_atm2_attn - raw_wind_attn - raw_refl1_attn - raw_refl2_attn)

 # Temperature in Celsius, multiplied by 100 for mozaic making
 temp = ((PB / (np.log(PR1 / (PR2 * (raw_obj + PO)) + PF)) - 273.15)* 100).astype(np.uint16)
 
 # Additional temperature correcting for groups of similiar images and export to TIFF
 list_of_temp.append(temp)
 if len(list_of_temp) < 24:
    temp = temp - 1398
    print('TEMPERATURE CALCULATED')
    out_tif_path =os.path.join(r'C:\Users\Terez\Desktop\skola\UP\BP\scripts\UNTITLED\output',csv_file2.replace('.csv', '.tif'))
    tifffile.imsave(out_tif_path, temp)
 
 if len(list_of_temp) > 23 and len(list_of_temp) < 31:
    temp = temp - 918
    print('TEMPERATURE CALCULATED')
    out_tif_path = os.path.join(r'C:\Users\Terez\Desktop\skola\UP\BP\scripts\UNTITLED\output',csv_file2.replace('.csv', '.tif'))
    tifffile.imsave(out_tif_path, temp)

     if len(list_of_temp) > 30 and len(list_of_temp) < 56 :
        temp = temp - 816
        print('TEMPERATURE CALCULATED')
        out_tif_path =os.path.join(r'C:\Users\Terez\Desktop\skola\UP\BP\scripts\UNTITLED\output',csv_file2.replace('.csv', '.tif'))
        tifffile.imsave(out_tif_path, temp)
     
     if len(list_of_temp) > 68 and len(list_of_temp) < 84:
        temp = temp - 1091
        print('TEMPERATURE CALCULATED')
        out_tif_path = os.path.join(r'C:\Users\Terez\Desktop\skola\UP\BP\scripts\UNTITLED\output',csv_file2.replace('.csv', '.tif'))
        tifffile.imsave(out_tif_path, temp)
     
     if len(list_of_temp) > 103:
        temp = temp - 943
        print('TEMPERATURE CALCULATED')
        out_tif_path =os.path.join(r'C:\Users\Terez\Desktop\skola\UP\BP\scripts\UNTITLED\output',csv_file2.replace('.csv', '.tif'))
        tifffile.imsave(out_tif_path, temp)
     else:
        print('TEMPERATURE CALCULATED')
        out_tif_path =os.path.join(r'C:\Users\Terez\Desktop\skola\UP\BP\scripts\UNTITLED\output',csv_file2.replace('.csv', '.tif'))
        tifffile.imsave(out_tif_path, temp)
 print('Finished')
 return temp
raw2temp()
