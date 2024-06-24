import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings("ignore")

demand_data = pd.read_csv('data\dataset\demand.csv')
vehicle_data = pd.read_csv('vehicle_data_edit.csv')
vehicle_fuels = pd.read_csv('data/dataset/vehicles_fuels.csv')
fuels_data = pd.read_csv('data/dataset/fuels.csv')
carbon_emissions_data = pd.read_csv('data/dataset/carbon_emissions.csv')
cost_profiles = pd.read_csv('data/dataset/cost_profiles.csv')

def feasible_vehicle(yr, s,d):
    temp_df = vehicle_data.loc[(vehicle_data['Year'] <= yr) & (vehicle_data['Size'] == s)]
    if d == 'D2':
        temp_df = temp_df.loc[temp_df['Distance'] != 'D1']
    elif d == 'D3':
        temp_df = temp_df.loc[(temp_df['Distance'] == 'D3') | (temp_df['Distance'] == 'D4')]
    elif d == 'D4':
        temp_df = temp_df.loc[(temp_df['Distance'] == 'D4')]
    return temp_df

def feasible_vehicle_buy(yr, s,d):
    temp_df = vehicle_data.loc[(vehicle_data['Year'] == yr) & (vehicle_data['Size'] == s)]
    if d == 'D2':
        temp_df = temp_df.loc[temp_df['Distance'] != 'D1']
    elif d == 'D3':
        temp_df = temp_df.loc[(temp_df['Distance'] == 'D3') | (temp_df['Distance'] == 'D4')]
    elif d == 'D4':
        temp_df = temp_df.loc[(temp_df['Distance'] == 'D4')]
    return temp_df

def decide_vehicle(feas_veh, prob):
    rand = random.random()
    for i in range(len(prob)):
        if rand < sum(prob[:i+1]):
            return feas_veh.iloc[i]
        
def feasible_from_avail_vehicle(avail_vehicle, feas_veh):
    new_feas_vehicle = []
    arr_distance = []
    for k in range(len(avail_vehicle['ID'])):
        i = avail_vehicle['ID'][k]
        if i in feas_veh['ID'].tolist():
            new_feas_vehicle.append(i)
            arr_distance.append(avail_vehicle['distance'][k])
    return {'ID': new_feas_vehicle, 'distance': arr_distance}

def sort_vehicle(vehicle_dict):
    list_BEV = []
    list_BEV_dist = []
    list_LNG = []
    list_LNG_dist = []
    list_diesel = []
    list_diesel_dist = []
    n = len(vehicle_dict['ID'])
    for k in range(n):
        i = vehicle_dict['ID'][0]
        temp = i.split('_')
        if temp[0] == 'BEV':
            list_BEV.append(i)
            list_BEV_dist.append(vehicle_dict['distance'][0])
        elif temp[0] == 'LNG':
            list_LNG.append(i)
            list_LNG_dist.append(vehicle_dict['distance'][0])
        else:
            list_diesel.append(i)
            list_diesel_dist.append(vehicle_dict['distance'][0])
        del vehicle_dict['ID'][0]
        del vehicle_dict['distance'][0]

    return {'ID': list_BEV + list_LNG + list_diesel, 'distance': list_BEV_dist + list_LNG_dist + list_diesel_dist}

def count_year_vehicle(list_vehicle):
    dict_year = {}
    for i in range(2023,2039):
        dict_year[i] = 0
        for j in range(len(list_vehicle)):
            veh = list_vehicle[j]
            year_buy = veh.split('_')
            year_buy = year_buy[-1]
            if year_buy == str(i):
                dict_year[i] += 1
    return dict_year

def func_cost_fuel(veh_type, vehicle_fuel, yearly_dist, year, prob_ef):
    if veh_type == 'LNG':
        #cost fuel non eco
        consumption_unit = vehicle_fuel.loc[(vehicle_fuel['Fuel'] == 'LNG')]
        consumption_unit = consumption_unit['Consumption (unit_fuel/km)'].iloc[0]
        cost_per_fuel = fuels_data.loc[(fuels_data['Fuel'] == 'LNG') & (fuels_data['Year'] == year)]
        cost_per_fuel = cost_per_fuel['Cost ($/unit_fuel)'].iloc[0]
        dist = (1-prob_ef)*yearly_dist
        cost_non_eco = consumption_unit*cost_per_fuel*dist

        #cost eco
        consumption_unit = vehicle_fuel.loc[(vehicle_fuel['Fuel'] == 'BioLNG')]
        consumption_unit = consumption_unit['Consumption (unit_fuel/km)'].iloc[0]
        cost_per_fuel = fuels_data.loc[(fuels_data['Fuel'] == 'BioLNG') & (fuels_data['Year'] == year)]
        cost_per_fuel = cost_per_fuel['Cost ($/unit_fuel)'].iloc[0]
        dist = (prob_ef)*yearly_dist
        cost_eco = consumption_unit*cost_per_fuel*dist
        # cost_fuel = cost_eco + cost_non_eco
    elif veh_type == 'Diesel':
        #cost fuel non eco
        consumption_unit = vehicle_fuel.loc[(vehicle_fuel['Fuel'] == 'B20')]
        consumption_unit = consumption_unit['Consumption (unit_fuel/km)'].iloc[0]
        cost_per_fuel = fuels_data.loc[(fuels_data['Fuel'] == 'B20') & (fuels_data['Year'] == year)]
        cost_per_fuel = cost_per_fuel['Cost ($/unit_fuel)'].iloc[0]
        dist = (1-prob_ef)*yearly_dist
        cost_non_eco = consumption_unit*cost_per_fuel*dist

        #cost eco
        consumption_unit = vehicle_fuel.loc[(vehicle_fuel['Fuel'] == 'HVO')]
        consumption_unit = consumption_unit['Consumption (unit_fuel/km)'].iloc[0]
        cost_per_fuel = fuels_data.loc[(fuels_data['Fuel'] == 'HVO') & (fuels_data['Year'] == year)]
        cost_per_fuel = cost_per_fuel['Cost ($/unit_fuel)'].iloc[0]
        dist = (prob_ef)*yearly_dist
        cost_eco = consumption_unit*cost_per_fuel*dist
        # cost_fuel = cost_eco + cost_non_eco
    else:
        cost_non_eco = 0
        consumption_unit = vehicle_fuel.loc[(vehicle_fuel['Fuel'] == 'Electricity')]
        consumption_unit = consumption_unit['Consumption (unit_fuel/km)'].iloc[0]
        cost_per_fuel = fuels_data.loc[(fuels_data['Fuel'] == 'Electricity') & (fuels_data['Year'] == year)]
        cost_per_fuel = cost_per_fuel['Cost ($/unit_fuel)'].iloc[0]
        dist = yearly_dist
        cost_eco = consumption_unit*cost_per_fuel*dist
        # cost_fuel = cost_eco + cost_non_eco
    return cost_eco + cost_non_eco
    
# def carbon_capture():

unique_year = np.unique(demand_data['Year'])
s = np.unique(demand_data['Size'])
d = np.unique(demand_data['Distance'])

# env_friendly_benchmark = (np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.6, 0.6, 0.6, 0.6, 0.4, 0.1, 0.1, 0.1, 0.1]) - 0.1).tolist()
coef = np.array([ 4.51451197e-04,  3.22017751e+00, -1.09844901e-01,  1.00008622e+00,
       -3.96969868e-04])

arr_cc_expected = carbon_emissions_data['Carbon emission CO2/kg'].tolist()

# dict_year = {}
# dict_year_sell = {}
# dict_year_buy = {}
# dict_total_fleet = {}
# for yr in unique_year:
#    dict_year[yr] = []
#    dict_year_sell[yr] = []
#    dict_year_buy[yr] = []
#    dict_total_fleet[yr] = {'ID': []}
def calculate_cost(coef_buy, coef_sell):
   try:
      avail_vehicle = {'ID': [],
                     'buy_cost': [],
                     'distance': []}
      total_fleet = {'ID': [],
                     'buy_cost': [],
                     'size': [],
                     'distance_ability': [],
                     'yearly_distance': []}

      buy = {}
      use = {}
      sell = {}
      arr_env_friendly = []
      arr_cc = []
      arr_cost_buy= []
      arr_cost_ins = []
      arr_cost_mnt = []
      arr_cost_fuel = []
      arr_total_cost = []
      arr_cost_sell = []
      arr_max_sold = []
      arr_sold = []
      # coef_buy = 0.195
      # coef_sell = 0.195

      for yr in unique_year:
         #buy
         buy[yr] = {'ID': [],
                  'cost': [],
                  'distance': []}
         use[yr] = {'ID': [],
                  'size': [],
                  'distance_bucket': [],
                  'distance': []
                  }
         # print(yr)
         for i in range(len(s)):
            demand_avail = 0
            for j in range(len(d)-1,-1,-1):
               feas_veh = feasible_vehicle(yr,s[i],d[j])
               new_feas_vehicle = feasible_from_avail_vehicle(avail_vehicle,feas_veh)
               sorted_feas_vehicle = sort_vehicle(new_feas_vehicle)
               demand = demand_data['Demand (km)'].loc[(demand_data['Size'] == s[i]) & (demand_data['Distance'] == d[j]) & (demand_data['Year'] == yr)].iloc[0]
               while demand > 0:
                  if (len(sorted_feas_vehicle['ID']) == 0) & (demand > 0):
                     feas_veh = feasible_vehicle_buy(yr,s[i],d[j])
                     fuel_cost = []
                     for idx in range(len(feas_veh)):
                        veh = feas_veh['ID'].iloc[idx]
                        veh_type = veh.split('_')
                        veh_type = veh_type[0]
                        temp_df = vehicle_fuels.loc[vehicle_fuels['ID'] == veh]
                        if yr == 2023:
                           fuel_cost.append(func_cost_fuel(veh_type,temp_df,feas_veh['Yearly range (km)'].iloc[idx],yr,0))
                        else:
                           fuel_cost.append(func_cost_fuel(veh_type,temp_df,feas_veh['Yearly range (km)'].iloc[idx],yr,arr_env_friendly[-1]-0.1))
                        # func_cost_fuel(veh_replace_type, temp_df,temp_df_decide['Yearly range (km)'],yr+1)
                     yearly_cost = coef_buy*feas_veh['Cost ($)'] + np.array(fuel_cost)
                     #0.265 = (cost + (total biaya ins+mnt selama 10 th) 1.95*cost - 0.3*cost (resale))/10 assume 1 kendaraan untuk 10 th
                     #0.19 = (cost + 0.27*cost (ins+mnt 3 tahun) - 0.7*cost)/3 assume 1 kendaraan untuk 3 tahun
                     feas_veh['yearly_cost'] = yearly_cost
                     inv = 1/feas_veh['yearly_cost']
                     inv[inv.idxmax()] = inv[inv.idxmax()]+1
                     # inv[inv.idxmax()] = inv[inv.idxmax()]*6
                     prob = (inv/sum(inv)).values
                     temp_df_decide = decide_vehicle(feas_veh, prob)
                     buy[yr]['ID'].append(temp_df_decide['ID'])
                     buy[yr]['distance'].append(temp_df_decide['Yearly range (km)'])
                     buy[yr]['cost'].append(temp_df_decide['Cost ($)'])
                     sorted_feas_vehicle['ID'].append(temp_df_decide['ID'])
                     sorted_feas_vehicle['distance'].append(temp_df_decide['Yearly range (km)'])
                     avail_vehicle['ID'].append(temp_df_decide['ID'])
                     avail_vehicle['buy_cost'].append(temp_df_decide['Cost ($)'])
                     avail_vehicle['distance'].append(temp_df_decide['Yearly range (km)'])
                  idx_used = np.argmin(np.abs(np.array(sorted_feas_vehicle['distance']) - demand))
                  demand0 = demand
                  demand = demand - sorted_feas_vehicle['distance'][idx_used]
                  use[yr]['ID'].append(sorted_feas_vehicle['ID'][idx_used])
                  use[yr]['size'].append(s[i])
                  use[yr]['distance_bucket'].append(d[j])
                  if demand < 0:
                     use[yr]['distance'].append(demand0)
                  else:
                     use[yr]['distance'].append(sorted_feas_vehicle['distance'][idx_used])

                  veh = sorted_feas_vehicle['ID'][idx_used]
                  idx = avail_vehicle['ID'].index(veh)
                  del avail_vehicle['ID'][idx]
                  del avail_vehicle['distance'][idx]
                  del avail_vehicle['buy_cost'][idx]

                  del sorted_feas_vehicle['ID'][idx_used]
                  del sorted_feas_vehicle['distance'][idx_used]
                  
         #after buy
         total_fleet['ID'] = total_fleet['ID'] + buy[yr]['ID']
         total_fleet['buy_cost'] = total_fleet['buy_cost'] + buy[yr]['cost']
         arr_size = []
         arr_distance_ability = []
         arr_yearly_distance = []
         for i in range(len(buy[yr]['ID'])):
            veh = buy[yr]['ID'][i]
            temp_df = vehicle_data.loc[vehicle_data['ID'] == veh]
            arr_size.append(temp_df['Size'].iloc[0])
            arr_distance_ability.append(temp_df['Distance'].iloc[0])
            arr_yearly_distance.append(temp_df['Yearly range (km)'].iloc[0])
         total_fleet['size'] = total_fleet['size'] + arr_size
         total_fleet['distance_ability'] = total_fleet['distance_ability'] + arr_distance_ability
         total_fleet['yearly_distance'] = total_fleet['yearly_distance'] + arr_yearly_distance
         
         #after use
         df_use = pd.DataFrame(use[yr])
         #hitung carbon capture
         cc_real = np.inf
         prob_env_friendly = 0.0
         while cc_real > arr_cc_expected[yr-2023]:
            cc = 0
            use[yr]['fuel'] = []
            for i in range(len(df_use)):
               rand_num = random.random()
               veh = df_use['ID'].iloc[i]
               dist = df_use['distance'].iloc[i]
               temp_df = vehicle_fuels.loc[vehicle_fuels['ID'] == veh]
               veh_type = veh.split('_')
               veh_type = veh_type[0]
               if veh_type == 'BEV':
                  fuel = 'Electricity'
               elif veh_type == 'LNG':
                  if rand_num > prob_env_friendly:
                     fuel = 'LNG'
                  else:
                     fuel = 'BioLNG'
               else:
                  if rand_num > prob_env_friendly:
                     fuel = 'B20'
                  else:
                     fuel = 'HVO'
               use[yr]['fuel'].append(fuel)
               consumption_unit = temp_df.loc[(temp_df['Fuel'] == fuel)]
               consumption_unit = consumption_unit['Consumption (unit_fuel/km)'].iloc[0]
               emission = fuels_data.loc[(fuels_data['Fuel'] == fuel) & (fuels_data['Year'] == yr)]
               emission = emission['Emissions (CO2/unit_fuel)'].iloc[0]
               cc_veh = consumption_unit*emission*dist
               cc = cc + cc_veh
            cc_real = cc
            prob_env_friendly += 0.1
            if (prob_env_friendly > 1) & (cc_real > arr_cc_expected[yr-2023]):
               print('exceed carbon tolerance, need revised available vehicle')
               break
         arr_cc.append(cc)
         arr_env_friendly.append(prob_env_friendly)

         #hitung cost
         #buy
         cost_buy = 0
         for i in buy[yr]['ID']:
            veh_cost = vehicle_data.loc[(vehicle_data['ID'] == i)]
            veh_cost = veh_cost['Cost ($)'].iloc[0]
            cost_buy = cost_buy + veh_cost

         #avail
         cost_ins = 0
         cost_mnt = 0
         for i in range(len(total_fleet['ID'])):
            veh = total_fleet['ID'][i]
            year_buy = veh.split('_')
            year_buy = int(year_buy[-1])
            delta_year = yr - year_buy + 1
            temp_df = cost_profiles.loc[cost_profiles['End of Year'] == delta_year]
            pct_ins = temp_df['Insurance Cost %'].iloc[0]/100
            pct_mnt = temp_df['Maintenance Cost %'].iloc[0]/100
            cost_ins = cost_ins + pct_ins*total_fleet['buy_cost'][i]
            cost_mnt = cost_mnt + pct_mnt*total_fleet['buy_cost'][i]

         #cost fuel
         total_cost_fuel = 0
         for i in range(len(use[yr]['ID'])):
            veh = use[yr]['ID'][i]
            dist = use[yr]['distance'][i]
            temp_df = vehicle_fuels.loc[vehicle_fuels['ID'] == veh]
            fuel = use[yr]['fuel'][i]
            consumption_unit = temp_df.loc[(temp_df['Fuel'] == fuel)]
            consumption_unit = consumption_unit['Consumption (unit_fuel/km)'].iloc[0]
            cost_per_fuel = fuels_data.loc[(fuels_data['Fuel'] == fuel) & (fuels_data['Year'] == yr)]
            cost_per_fuel = cost_per_fuel['Cost ($/unit_fuel)'].iloc[0]
            total_cost_fuel = total_cost_fuel + consumption_unit*cost_per_fuel*dist

         total_cost = coef[0]*cost_buy + coef[1]*cost_ins + coef[2]*cost_mnt + coef[3]*total_cost_fuel
         arr_cost_buy.append(cost_buy)
         arr_cost_fuel.append(total_cost_fuel)
         arr_cost_ins.append(cost_ins)
         arr_cost_mnt.append(cost_mnt)
         # arr_total_cost.append(total_cost)
         # dict_total_fleet[yr]['ID'] = total_fleet['ID'].copy()
         #sell
         max_sold = int(0.2*len(total_fleet['ID']))
         arr_max_sold.append(max_sold)
         df_total_fleet = pd.DataFrame(total_fleet)
         if yr < 2038:
            next_ins = []
            next_mnt = []
            resale_decline = []
            arr_penalty = []
            next_fuel = []
            for i in range(len(df_total_fleet)):
               veh = df_total_fleet['ID'].iloc[i]
               year_buy = veh.split('_')
               year_buy = int(year_buy[-1])
               delta_year = yr - year_buy + 1
               temp_df = cost_profiles.loc[cost_profiles['End of Year'] <= delta_year+1]
               pct_ins = temp_df['Insurance Cost %'].iloc[-1]/100
               pct_mnt = temp_df['Maintenance Cost %'].iloc[-1]/100
               pct_rd = -(temp_df['Resale Value %'].iloc[-1] - temp_df['Resale Value %'].iloc[-2])/100
               temp_df = vehicle_fuels.loc[vehicle_fuels['ID'] == veh]
               # fuel = use[yr]['fuel'][i]
               veh_type = veh.split('_')
               veh_type = veh_type[0]
               if yr+1 > 2038:
                  cost_fuel = func_cost_fuel(veh_type, temp_df,df_total_fleet['yearly_distance'].iloc[i],yr,arr_env_friendly[-1])
               else:
                  cost_fuel = func_cost_fuel(veh_type, temp_df,df_total_fleet['yearly_distance'].iloc[i],yr+1,arr_env_friendly[-1])

               next_ins.append(pct_ins*df_total_fleet['buy_cost'].iloc[i])
               next_mnt.append(pct_mnt*df_total_fleet['buy_cost'].iloc[i])
               resale_decline.append(pct_rd*df_total_fleet['buy_cost'].iloc[i])
               next_fuel.append(cost_fuel)
               arr_penalty.append(1+0.15*(delta_year/10)**4)
               # arr_penalty.append(1)

            df_total_fleet['next_ins'] = np.array(next_ins)
            df_total_fleet['next_mnt'] = np.array(next_mnt)
            df_total_fleet['resale_decline'] = np.array(resale_decline)
            df_total_fleet['next_fuel'] = np.array(next_fuel)
            df_total_fleet['next_cost'] = df_total_fleet['next_ins'] + df_total_fleet['next_mnt'] + df_total_fleet['resale_decline'] + df_total_fleet['next_fuel']
            df_total_fleet['cost_penalty'] = df_total_fleet['next_cost']*np.array(arr_penalty)
         else:
            resale_price = []
            for i in range(len(df_total_fleet)):
               veh = df_total_fleet['ID'].iloc[i]
               year_buy = veh.split('_')
               year_buy = int(year_buy[-1])
               delta_year = yr - year_buy + 1
               temp_df = cost_profiles.loc[cost_profiles['End of Year'] <= delta_year+1]
               # pct_ins = temp_df['Insurance Cost %'].iloc[-1]/100
               # pct_mnt = temp_df['Maintenance Cost %'].iloc[-1]/100
               pct_rd = (temp_df['Resale Value %'].iloc[-1])/100
               resale_price.append(pct_rd*df_total_fleet['buy_cost'].iloc[i])
            df_total_fleet['cost_penalty'] = resale_price

         sorted_df_total_fleet = df_total_fleet.sort_values(by=['cost_penalty'], ascending=False)
         sell_threshold = max_sold
         if yr < 2038:
            #compare dengan vehicle di tahun depannya
            arr_replace_cost = []
            for i in range(len(sorted_df_total_fleet)):
               veh = sorted_df_total_fleet['ID'].iloc[i]
               #cek distance bucket
               dist_bucket = vehicle_data.loc[(vehicle_data['ID'] == veh)]
               dist_bucket = dist_bucket['Distance'].iloc[0]
               size = veh.split('_')
               size = size[1]
               feas_veh = feasible_vehicle_buy(yr+1,size,dist_bucket)
               yearly_cost = coef_buy*feas_veh['Cost ($)'] + feas_veh['fuel_cost']
               #0.265 = (cost + (total biaya ins+mnt selama 10 th) 1.95*cost - 0.3*cost (resale))/10 assume 1 kendaraan untuk 10 th
               #0.19 = (cost + 0.27*cost (ins+mnt 3 tahun) - 0.7*cost)/3 assume 1 kendaraan untuk 3 tahun
               feas_veh['yearly_cost'] = yearly_cost
               inv = 1/feas_veh['yearly_cost']
               inv[inv.idxmax()] = inv[inv.idxmax()]+1
               prob = (inv/sum(inv)).values
               temp_df_decide = decide_vehicle(feas_veh, prob)
               veh_replace = temp_df_decide['ID']
               cost_replace = temp_df_decide['Cost ($)']
               veh_replace_type = veh_replace.split('_')
               veh_replace_type = veh_replace_type[0]
               temp_df = vehicle_fuels.loc[vehicle_fuels['ID'] == veh_replace]
               cost_fuel = func_cost_fuel(veh_replace_type, temp_df,temp_df_decide['Yearly range (km)'],yr+1,arr_env_friendly[-1])
               arr_replace_cost.append(coef_sell*cost_replace + cost_fuel)
            sorted_df_total_fleet['replacement_cost'] = arr_replace_cost
            sorted_df_total_fleet['differences'] = sorted_df_total_fleet['cost_penalty'] - sorted_df_total_fleet['replacement_cost']

            sorted_df_total_fleet = sorted_df_total_fleet.sort_values(by=['differences'], ascending=False)
            idx_positive = len(sorted_df_total_fleet.loc[(sorted_df_total_fleet['differences'] >= 0)])
            sell_threshold = min([max_sold, idx_positive])
      #    print(sorted_df_total_fleet)
         arr_sold.append(sell_threshold)
         sell[yr] = {'ID': [],
                     'cost_sell': []}

         sell[yr]['ID'] = sorted_df_total_fleet['ID'].iloc[:sell_threshold].tolist()
         cost_sell_per_veh = []
         for i in range(len(sell[yr]['ID'])):
            veh = sell[yr]['ID'][i]
            year_buy = veh.split('_')
            year_buy = int(year_buy[-1])
            delta_year = yr - year_buy + 1
            temp_df = cost_profiles.loc[cost_profiles['End of Year'] == delta_year]
            pct_resale = temp_df['Resale Value %'].iloc[0]/100
            cost_sell_per_veh.append(sorted_df_total_fleet['buy_cost'].iloc[i]*pct_resale)
         sell[yr]['cost_sell'] = cost_sell_per_veh

         cost_sell = sum(cost_sell_per_veh)
         arr_cost_sell.append(cost_sell)

         #calculate fleet availability after sell
         avail_vehicle['ID'] = sorted_df_total_fleet['ID'].iloc[sell_threshold:].tolist()
         avail_vehicle['buy_cost'] = sorted_df_total_fleet['buy_cost'].iloc[sell_threshold:].tolist()
         arr_dist = []
         for i in range(len(avail_vehicle['ID'])):
            veh = avail_vehicle['ID'][i]
            temp_df = vehicle_data.loc[(vehicle_data['ID'] == veh)]
            arr_dist.append(temp_df['Yearly range (km)'].iloc[0])
         avail_vehicle['distance'] = arr_dist

         total_fleet['ID'] = avail_vehicle['ID'].copy()
         total_fleet['buy_cost'] = avail_vehicle['buy_cost'].copy()
         total_fleet['yearly_distance'] = avail_vehicle['distance'].copy()
         arr_size = []
         arr_distance_ability = []
         for i in range(len(total_fleet['ID'])):
            veh = total_fleet['ID'][i]
            temp_df = vehicle_data.loc[vehicle_data['ID'] == veh]
            arr_size.append(temp_df['Size'].iloc[0])
            arr_distance_ability.append(temp_df['Distance'].iloc[0])
         total_fleet['size'] = arr_size
         total_fleet['distance_ability'] = arr_distance_ability

         total_cost = total_cost + coef[4]*cost_sell
         arr_total_cost.append(total_cost)
   except IndexError:
      arr_total_cost = [np.inf for i in unique_year]
      print('constrain maximum vehicle lifespan is violated')
   return arr_total_cost
   # counter = count_year_vehicle(total_fleet['ID'])
   # for i in counter.keys():
   #    dict_year[i].append(counter[i])

   # counter_buy = count_year_vehicle(buy[yr]['ID'])
   # for i in counter_buy.keys():
   #    dict_year_buy[i].append(counter_buy[i])

   # counter_sell = count_year_vehicle(sell[yr]['ID'])
   # for i in counter_sell.keys():
   #    dict_year_sell[i].append(counter_sell[i])

def check_boundaries(x):
    logic1 = False
    logic2 = False
    logic3 = False
    logic4 = False
    if x[0] >= 0.01:
        logic1 = True
    if x[1] >= 0.16:
        logic2 = True 
    if x[0] <= 0.50777776:
        logic3 = True 
    if x[1] <= 0.27:
        logic4 = True
    return logic1 and logic2 and logic3 and logic4 
m = 10
kmax = 10
n = 2

arr_coef_buy = np.random.rand(1,m)*(0.50777776 - 0.01) + 0.01
arr_coef_sell = np.random.rand(1,m)*(0.27 - 0.16) + 0.16
r = 0.95
theta = np.pi/4
Un = np.eye(n)
for i in range(n - 1):
    Qn = np.eye(n)
    for j in range(i+1):
        Rij = np.eye(n)
        Rij[n - (i+2), n - (i+2)] = np.cos(theta)
        Rij[n - (i+2), n + 1 - (j+2)] = -np.sin(theta)
        Rij[n + 1 - (j+2), n - (i+2)] = np.sin(theta)
        Rij[n + 1 - (j+2), n + 1 - (j+2)] = np.cos(theta)
        Pn = Qn @ Rij # np.dot(Qn, Rij)
        Qn = Pn
    Tn = Un @ Pn #np.dot(Un, Pn)
    Un = Tn
Rn = Un
Sn = r * Rn

x = np.concatenate((arr_coef_buy, arr_coef_sell),axis=0)

arr_private_cost = []
arr_public_cost = []
for i in range(m):
    print(i)
    coef_buy = arr_coef_buy[0,i]
    coef_sell = arr_coef_sell[0,i]
    arr_cost = calculate_cost(coef_buy, coef_sell)
    total_cost = sum(arr_cost)
    arr_public_cost.append(total_cost)
    arr_private_cost.append(sum(arr_cost[:6]))

idxmin = np.argmin(arr_public_cost)
param = x[:,idxmin:idxmin+1]
print('private cost: ', arr_private_cost[idxmin])

dict_public_cost = {}
dict_private_cost = {}
dict_x = {}
dict_x[0] = x
dict_private_cost[0] = arr_private_cost
dict_public_cost[0] = arr_public_cost
arr_param = [param.tolist()]

for it in range(kmax):
    print('it: ', it)
    xstar = np.tile(param.reshape(-1,1), (1, m))
    px = Sn @ x - (Sn - np.eye(n)) @ xstar
    x_old = x
    x = px
    for j in range(m):
        is_feasible = check_boundaries(x[:,j])
        if is_feasible:
            arr_cost = calculate_cost(x[0,j], x[1,j])
            arr_private_cost[j] = sum(arr_cost[:6])
            arr_public_cost[j] = sum(arr_cost)
        else:
            arr_private_cost[j] = np.inf
            arr_public_cost[j] = np.inf
    idxmin = np.argmin(arr_public_cost)
    print('private cost: ', arr_private_cost[idxmin])
    param = x[:,idxmin:idxmin+1]
    dict_private_cost[it+1] = arr_private_cost
    dict_public_cost[it+1] = arr_public_cost
    dict_x[it+1] = x
    arr_param.append(param.tolist())