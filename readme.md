# Shell.ai Hackathon 2024 Solution

This repository contains my solution for the Shell.ai Hackathon 2024, where I successfully advanced to Level 2. The challenge involved developing innovative solutions to real-world energy and sustainability problems using AI and data science techniques.

## Overview

### Problem Statement
The hackathon presented a problem focused on optimizing energy management systems using AI-driven approaches. The goal was to enhance efficiency, reduce costs, and promote sustainability in energy supply chains. The problem statement can be accessed here: [Shell.ai Hackathon 2024 Problem Statement](https://www.hackerearth.com/challenges/new/competitive/shellai-hackathon-2024/)

### Solution Approach
To address the problem, I simplified the decision-making process for vehicle planning by introducing two key variables: `coef_buy` and `coef_sell`. These variables act as decision parameters, allowing for a more straightforward optimization. Instead of managing a fleet of vehicles and their complex planning, the approach reduces the problem to finding optimal scalar values for `coef_buy` and `coef_sell`. This simplification aims to minimize the objective function more efficiently, focusing on the core variables that drive the decision-making process.

The solution is implemented in the `calculate_cost` function, with the following explanation:

#### Input:
- **Demand Data**: Historical and projected demand for vehicle types.
- **Vehicle Data**: Information on available vehicles, including costs, types (LNG, Diesel, BEV), and features.
- **Cost Data**: Details on buying, maintaining, and operating costs.
- **Environmental Data**: Emissions and carbon capture information for different vehicle types.
- **Fleet Data**: Current fleet composition and associated costs.

#### Output:
- **Updated Fleet Data**: Changes in vehicle fleet composition after processing.
- **Cost Reports**: Breakdown of costs, including buying, maintenance, and fuel expenses.
- **Replacement Decisions**: Recommendations for vehicle purchases and sales.

#### Steps:

1. **Initialize Variables**
   - Set up variables to track total costs, fleet status, and other relevant metrics.

2. **Process Demand Data**
   - Evaluate the demand for vehicles based on historical and projected data for each year.

3. **Determine Vehicle Selection**
   - Identify available vehicles that meet demand criteria.
   - Evaluate each vehicle based on cost, fuel type, and other factors.

4. **Calculate Buying Costs**
   - Compute:
     - Purchase cost.
     - Maintenance costs.
     - Fuel costs based on vehicle type and usage patterns.

5. **Calculate Usage Costs**
   - Calculate:
     - Fuel consumption.
     - Carbon emissions and environmental impact.
     - Insurance and maintenance costs.

6. **Estimate Resale Values**
   - Estimate potential resale values for vehicles that may be sold.

7. **Decide on Vehicle Purchases and Sales**
   - Determine which vehicles to buy based on cost-effectiveness and environmental impact.
   - Decide which vehicles to sell based on resale value and overall cost-effectiveness.

8. **Update Fleet Data**
   - Adjust fleet composition to reflect new purchases and sales.
   - Update cost records for the current year.

9. **Generate Cost Reports**
   - Produce reports on:
     - Total buying costs.
     - Maintenance and insurance costs.
     - Fuel expenses.
     - Carbon emissions and capture.

10. **Review and Adjust**
    - Assess results of the fleet management strategy.
    - Adjust parameters as needed for future iterations.

### Optimization with Spiral Dynamics

To find more optimal solutions, I employ a meta-heuristic Spiral Dynamics Optimization method to determine the optimal values for `coef_buy` and `coef_sell`. The process involves:

1. Generating `m` random values for `coef_buy` and `coef_sell`.
2. Running the `calculate_cost` function using these values.
3. Calculating new `coef_buy` and `coef_sell` based on Spiral Dynamics.
4. Repeating steps 2 and 3 until the maximum number of iterations (`kmax`) is reached.
5. Obtaining the optimal values for `coef_buy` and `coef_sell`, and refining the fleet planning.
