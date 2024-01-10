import pandas as pd   #  import Pandas for data manipulation, NumPy for numerical operations, and Matplotlib for plotting.
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data8.csv', header=None, names=['salary']) #pd.read is used to read the data  

salary_data = data['salary'] #  the 'salary' column from the DataFrame and stores it in the variable salary_data
mean_salary = np.mean(salary_data) # calculates the mean and standard deviation of the salary values using NumPy's 
std_dev_salary = np.std(salary_data) 
num_bins = 30  
counts, bin_edges = np.histogram(salary_data, bins=num_bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

#Divides the salary data into 30 bins, calculates the counts and bin edges using np.histogram(), and computes the bin centers for plotting

#Creates a figure with a size of 8x6 inches and plots a histogram of the salary data with 30 bins, setting transparency to 0.6, and assigning a label.

plt.figure(figsize=(8, 6))
plt.hist(salary_data, bins=num_bins, density=True, alpha=0.6, label='Salary Histogram')
plt.plot(bin_centers, counts, label='Probability Density Function', color='red')
plt.axvline(mean_salary, color='green', linestyle='dashed', linewidth=1.5, label=f'Mean Salary: {mean_salary:.2f}')
plt.axvline(mean_salary + std_dev_salary, color='orange', linestyle='dashed', linewidth=1.5,
            label=f'Std Dev: {std_dev_salary:.2f}') #Adds vertical dashed lines representing the mean and mean + standard deviation values to the plot, labeling them accordingly

#Sets labels for the x and y axes, a title for the plot, and displays the legend

plt.xlabel('Salary')
plt.ylabel('Probability Density') 
plt.title('Distribution of Salaries')
plt.legend()

#Saves the generated plot as an image file named 'salary_distribution_with_values.png' and displays the plot on the screen

plt.text(0.5, 0.95, f'Std Dev: {std_dev_salary:.2f}', transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top')

plt.savefig('salary_distribution_with_values.png') # Save the fig in the given name

plt.show() # Plot show

#Prints the calculated mean and standard deviation values
print(f"Mean Salary: {mean_salary:.2f}")
print(f"Standard Deviation: {std_dev_salary:.2f}")

