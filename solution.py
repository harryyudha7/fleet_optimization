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

def separate_ID(df_vehicle):
    arr_veh = []
    arr_year = []
    arr_size = []
    
# def carbon_capture():

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

unique_year = np.unique(demand_data['Year'])
s = np.unique(demand_data['Size'])
d = np.unique(demand_data['Distance'])

arr_cc = []
arr_cost_buy= []
arr_cost_ins = []
arr_cost_mnt = []
arr_cost_fuel = []
arr_total_cost = []
arr_cost_sell = []

arr_cc_expected = carbon_emissions_data['Carbon emission CO2/kg'].tolist()

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
   print(yr)
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
               yearly_cost = 0.12*feas_veh['Cost ($)'] + feas_veh['fuel_cost']
               feas_veh['yearly_cost'] = yearly_cost
               inv = 1/feas_veh['yearly_cost']
               inv[inv.idxmax()] = inv[inv.idxmax()]*6
               prob = (inv/sum(inv)).values
               temp_df_decide = decide_vehicle(feas_veh, prob)
               buy[yr]['ID'].append(temp_df_decide['ID'])
               buy[yr]['distance'].append(temp_df_decide['Yearly range (km)'])
               buy[yr]['cost'].append(temp_df_decide['Cost ($)'])
               # temp_dict = {'ID': temp_df_decide['ID'],
               #             'buy_cost': temp_df_decide['Cost ($)'],
               #             'size': s[i],
               #             'distance_ability': d[j],
               #             'yearly_distance': temp_df_decide['Yearly range (km)']}
               sorted_feas_vehicle['ID'].append(temp_df_decide['ID'])
               # sorted_feas_vehicle['buy_cost'].append(temp_dict['buy_cost'])
               # sorted_feas_vehicle['size'].append(temp_dict['size'])
               # sorted_feas_vehicle['distance_ability'].append(temp_dict['distance_ability'])
               sorted_feas_vehicle['distance'].append(temp_df_decide['Yearly range (km)'])
               avail_vehicle['ID'].append(temp_df_decide['ID'])
               avail_vehicle['buy_cost'].append(temp_df_decide['Cost ($)'])
               avail_vehicle['distance'].append(temp_df_decide['Yearly range (km)'])
            idx_used = np.argmin(np.abs(np.array(sorted_feas_vehicle['distance']) - demand))
            demand = demand - sorted_feas_vehicle['distance'][idx_used]
            use[yr]['ID'].append(sorted_feas_vehicle['ID'][idx_used])
            use[yr]['size'].append(s[i])
            use[yr]['distance_bucket'].append(d[j])
            use[yr]['distance'].append(sorted_feas_vehicle['distance'][idx_used])

            # sorted_feas_vehicle = sorted_feas_vehicle.drop([idx_used])

            veh = sorted_feas_vehicle['ID'][idx_used]
            idx = avail_vehicle['ID'].index(veh)
            del avail_vehicle['ID'][idx]
            del avail_vehicle['distance'][idx]
            del avail_vehicle['buy_cost'][idx]

            # temp_list = [0 for i in range(len(avail_vehicle['distance']))]
            # temp_list[idx] = sorted_feas_vehicle['distance'][idx_used]

            # avail_vehicle['distance'] = np.array(avail_vehicle['distance']) - np.array(temp_list)
            # is_zero = np.argwhere(abs(avail_vehicle['distance']) <= 0.01).tolist()
            # if len(is_zero) == 0:
            #    avail_vehicle['distance'] = avail_vehicle['distance'].tolist()
            # else:
            #    avail_vehicle['distance'] = avail_vehicle['distance'].tolist()
            #    del avail_vehicle['ID'][is_zero[0][0]]
            #    del avail_vehicle['distance'][is_zero[0][0]]
            #    del avail_vehicle['buy_cost'][is_zero[0][0]]

            del sorted_feas_vehicle['ID'][idx_used]
            # del sorted_feas_vehicle['buy_cost'][idx_used]
            # del sorted_feas_vehicle['size'][idx_used]
            # del sorted_feas_vehicle['distance_ability'][idx_used]
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
               print('rl')
               fuel = 'BioLNG'
         else:
            if rand_num > prob_env_friendly:
               fuel = 'B20'
            else:
               print('rl')
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
   # use[yr]['fuel'] = []
   total_cost_fuel = 0
   for i in range(len(use[yr]['ID'])):
      veh = use[yr]['ID'][i]
      dist = use[yr]['distance'][i]
      temp_df = vehicle_fuels.loc[vehicle_fuels['ID'] == veh]
      # veh_type = veh.split('_')
      # veh_type = veh_type[0]
      fuel = use[yr]['fuel'][i]
      # if veh_type == 'BEV':
      #    fuel = 'Electricity'
      # elif veh_type == 'LNG':
      #    fuel = 'LNG'
      # else:
      #    fuel = 'B20'
      # use[yr]['fuel'].append(fuel)
      consumption_unit = temp_df.loc[(temp_df['Fuel'] == fuel)]
      consumption_unit = consumption_unit['Consumption (unit_fuel/km)'].iloc[0]
      cost_per_fuel = fuels_data.loc[(fuels_data['Fuel'] == fuel) & (fuels_data['Year'] == yr)]
      cost_per_fuel = cost_per_fuel['Cost ($/unit_fuel)'].iloc[0]
      total_cost_fuel = total_cost_fuel + consumption_unit*cost_per_fuel*dist

   total_cost = cost_buy + cost_ins + cost_mnt + total_cost_fuel
   arr_cost_buy.append(cost_buy)
   arr_cost_fuel.append(total_cost_fuel)
   arr_cost_ins.append(cost_ins)
   arr_cost_mnt.append(cost_mnt)
   # arr_total_cost.append(total_cost)

   #sell
   max_sold = int(0.2*len(total_fleet['ID']))
   df_total_fleet = pd.DataFrame(total_fleet)

   next_ins = []
   next_mnt = []
   arr_penalty = []
   for i in range(len(df_total_fleet)):
      veh = df_total_fleet['ID'].iloc[i]
      year_buy = veh.split('_')
      year_buy = int(year_buy[-1])
      delta_year = yr - year_buy + 1
      temp_df = cost_profiles.loc[cost_profiles['End of Year'] == delta_year]
      pct_ins = temp_df['Insurance Cost %'].iloc[0]/100
      pct_mnt = temp_df['Maintenance Cost %'].iloc[0]/100
      next_ins.append(pct_ins*df_total_fleet['buy_cost'].iloc[i])
      next_mnt.append(pct_mnt*df_total_fleet['buy_cost'].iloc[i])
      arr_penalty.append(1+2*(delta_year/10)**4)

   df_total_fleet['next_ins'] = np.array(next_ins)
   df_total_fleet['next_mnt'] = np.array(next_mnt)
   df_total_fleet['next_cost'] = df_total_fleet['next_ins'] + df_total_fleet['next_mnt']
   df_total_fleet['cost_penalty'] = df_total_fleet['next_cost']*np.array(arr_penalty)

   sorted_df_total_fleet = df_total_fleet.sort_values(by=['cost_penalty'], ascending=False)
#    print(sorted_df_total_fleet)
   sell[yr] = {'ID': [],
               'cost_sell': []}

   sell[yr]['ID'] = sorted_df_total_fleet['ID'].iloc[:max_sold].tolist()
   cost_sell_per_veh = []
   for i in range(len(sell[yr]['ID'])):
      veh = df_total_fleet['ID'].iloc[i]
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
   avail_vehicle['ID'] = sorted_df_total_fleet['ID'].iloc[max_sold:].tolist()
   avail_vehicle['buy_cost'] = sorted_df_total_fleet['buy_cost'].iloc[max_sold:].tolist()
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

   total_cost = total_cost - cost_sell
   arr_total_cost.append(total_cost)