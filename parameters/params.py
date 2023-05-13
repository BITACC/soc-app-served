period1 = 50
period2 = 200
cols_for_training = [
    "Date",
    "Time",
    "Temperature",
    "pH",
    "DO",
    "Conductivity at 25",    
    "ORP",
    "Meas_Rate",
]
cols_for_training_except_date_time = [
    "Temperature",
    "pH",
    "DO",
    "Conductivity at 25",    
    "ORP",
    "Meas_Rate",
]
cols_for_training_except_date_time_CorrosionRate = [
    "Temperature",
    "pH",
    "DO",
    "Conductivity at 25",    
    "ORP",
]
target = 'Meas_Rate'

HZS = [
"Temperature",
"pH",
#"CO2",
"Dissolved oxygen [mg/l]",
#"Dissolved oxygen [%]",
#"Conductivity",
"Conductivity at 25°C",
#"Salinity",
#"Resistivity",
"ORP Redox potential",
#"Seawater Specific Gravity",
"Chloride [mg/l]",
#"Nitrate",
#"Sulphur ions",
#"Biofilm",
"Chlorophyll [µg/l]",
#"Turbidity",
#"TDS (Total Dissolved Solids)",
#"Corrosion rate (LPR)",
"Corrosion rate (LPR)",
#"Corrosion rate (LPR)",
#"Pitting indication LPR",
#"Pitting indication LPR",
#"Pitting indication LPR",
#"Metal Loss in A (ER)",
#"Metal Loss in A (ER)",
#"Metal Loss in A (ER)",
#"Metal loss in um"
]

UGent = [
"Temperature",
"pH",
#"CO2",
"Dissolved oxygen [mg/l]",
#"Dissolved oxygen [%]",
#"Conductivity",
"Conductivity at 25°C",
#"Salinity",
#"Resistivity",
"ORP Redox potential",
#"Seawater Specific Gravity",
#"Chloride [mg/l]",
#"Nitrate",
#"Sulphur ions",
#"Biofilm",
#"Chlorophyll [µg/l]",
#"Turbidity",
#"TDS (Total Dissolved Solids)",
#"Corrosion rate (LPR)",
"Corrosion rate (LPR)",
#"Corrosion rate (LPR)",
#"Pitting indication LPR",
#"Pitting indication LPR",
#"Pitting indication LPR",
#"Metal Loss in A (ER)",
#"Metal Loss in A (ER)",
#"Metal Loss in A (ER)",
#"Metal loss in um"
]

OCAS = [
"Temperature",
"pH",
#"CO2",
"Dissolved oxygen [mg/l]",
#"Dissolved oxygen [%]",
#"Conductivity",
"Conductivity at 25°C",
#"Salinity",
#"Resistivity",
"ORP Redox potential",
#"Seawater Specific Gravity",
"Chloride [mg/l]",
#"Nitrate",
#"Sulphur ions",
#"Biofilm",
#"Chlorophyll [µg/l]",
#"Turbidity",
#"TDS (Total Dissolved Solids)",
#"Corrosion rate (LPR)",
"Corrosion rate (LPR)",
#"Corrosion rate (LPR)",
#"Pitting indication LPR",
#"Pitting indication LPR",
#"Pitting indication LPR",
#"Metal Loss in A (ER)",
#"Metal Loss in A (ER)",
#"Metal Loss in A (ER)",
#"Metal loss in um"
]

KUL = [
"Temperature",
"pH",
#"CO2",
"Dissolved oxygen [mg/l]",
#"Dissolved oxygen [%]",
#"Conductivity",
"Conductivity at 25°C",
#"Salinity",
#"Resistivity",
"ORP Redox potential",
#"Seawater Specific Gravity",
#"Chloride [mg/l]",
#"Nitrate",
#"Sulphur ions",
#"Biofilm",
#"Chlorophyll [µg/l]",
"Turbidity",
#"TDS (Total Dissolved Solids)",
#"Corrosion rate (LPR)",
"Corrosion rate (LPR)",
#"Corrosion rate (LPR)",
#"Pitting indication LPR",
#"Pitting indication LPR",
#"Pitting indication LPR",
#"Metal Loss in A (ER)",
#"Metal Loss in A (ER)",
#"Metal Loss in A (ER)",
#"Metal loss in um"
]

# Model 1: HZS/UGENT (Coastal)      Model 2: KUL (Waste Water)  
# and  Model 3: OCAS (BASF, Drinking Process water)
models_columns = {'model1': HZS, 'model2': KUL, 'model3': OCAS }
models_description = [   {'label':  'Marine/Coastal (HZS, UGent)', 'value':1},
                         {'label':  'Waster Water Treatment (KUL)' , 'value': 2},
                         {'label':  'Drinking/Process Water (OCAS, BASF)', 'value': 3}
                    ]
ai_models = ['TimeDistributed', 'LSTM', 'MVLR', 'GRU', 'DNN', 'CNN']
ai_model = 1