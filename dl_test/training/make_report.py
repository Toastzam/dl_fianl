import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns # Seaborn을 추가하여 더 예쁜 그래프를 그릴 수 있습니다.

# --- Actual Image Folder Path ---
base_image_folder_path = r'C:\dl_final\dl_fianl\dl_test\training\images'

# --- Check if the folder exists ---
if not os.path.isdir(base_image_folder_path):
    print(f"Error: Image folder not found at: {base_image_folder_path}")
    print("Please check the path or prepare the images in this folder.")
    exit()

print(f"Scanning image folder: {base_image_folder_path}")

# --- Count images in each breed folder ---
image_counts = {}
dog_breeds_folders_found = []

# Iterate through all items in the base image folder
for item_name in os.listdir(base_image_folder_path):
    full_item_path = os.path.join(base_image_folder_path, item_name)

    # Process only if it's a directory (breed folder)
    if os.path.isdir(full_item_path):
        dog_breeds_folders_found.append(item_name) # Add found breed folder name

        # Count files inside the breed folder
        count = 0
        for file_name in os.listdir(full_item_path):
            if os.path.isfile(os.path.join(full_item_path, file_name)):
                count += 1
        image_counts[item_name] = count

if not dog_breeds_folders_found:
    print(f"Warning: No breed folders found in '{base_image_folder_path}'.")
    print("Please ensure your folder structure is like 'images/nXXXXXX-breed_name/'.")
    exit()

# Extract only breed names (remove nXXXXXX- prefix)
# FIXED: Use 'k' (folder name) instead of 'v' (image count) for splitting
breed_names_only = {k: k.split('-')[1] if '-' in k else k for k, v in image_counts.items()}
counts_list = [image_counts[folder_name] for folder_name in dog_breeds_folders_found]

# Create a Pandas DataFrame for easier manipulation and sorting
df_counts = pd.DataFrame({
    'Folder Name': dog_breeds_folders_found,
    'Breed Name': [breed_names_only[f] for f in dog_breeds_folders_found],
    'Image Count': counts_list
})

# Sort by image count in descending order for better visualization
df_counts_sorted = df_counts.sort_values(by='Image Count', ascending=False)


# --- Data Analysis Results Output (English) ---
total_breeds = len(df_counts_sorted)
total_images = df_counts_sorted['Image Count'].sum()
print(f"\n--- Data Analysis Results Summary ---")
print(f"Total Number of Breeds: {total_breeds}")
print(f"Total Number of Images: {total_images}")

if total_breeds > 0:
    min_count = df_counts_sorted['Image Count'].min()
    max_count = df_counts_sorted['Image Count'].max()
    avg_count = df_counts_sorted['Image Count'].mean()
    median_count = df_counts_sorted['Image Count'].median()
    std_dev = df_counts_sorted['Image Count'].std()

    print(f"Min Images per Breed: {min_count}")
    print(f"Max Images per Breed: {max_count}")
    print(f"Average Images per Breed: {avg_count:.2f}")
    print(f"Median Images per Breed: {median_count}")
    print(f"Standard Deviation of Image Counts: {std_dev:.2f}")


    print(f"\nBreeds with Min Image Count ({min_count} images):")
    min_breeds = df_counts_sorted[df_counts_sorted['Image Count'] == min_count]['Breed Name'].tolist()
    if len(min_breeds) > 5: # Limit list if too long
        print(min_breeds[:5], f"... (and {len(min_breeds)-5} more)")
    else:
        print(min_breeds)


    print(f"\nBreeds with Max Image Count ({max_count} images):")
    max_breeds = df_counts_sorted[df_counts_sorted['Image Count'] == max_count]['Breed Name'].tolist()
    if len(max_breeds) > 5: # Limit list if too long
        print(max_breeds[:5], f"... (and {len(max_breeds)-5} more)")
    else:
        print(max_breeds)
else:
    print("No image counts were collected. The folder might be empty or contain no files.")


# --- Visualization 1: Horizontal Bar Chart (All Breeds, Sorted by Count) ---
plt.figure(figsize=(15, total_breeds * 0.3)) # Dynamic height based on number of breeds
plt.barh(df_counts_sorted['Breed Name'], df_counts_sorted['Image Count'], color=sns.color_palette("viridis", total_breeds)) # Use a color palette
plt.xlabel('Number of Images')
plt.ylabel('Dog Breed')
plt.title('Distribution of Image Counts per Dog Breed (Sorted)')
plt.gca().invert_yaxis() # Invert y-axis to show highest count at the top
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


# --- Visualization 2: Pie Chart (Top N + Others) ---
top_n_pie = min(15, total_breeds) # Show top 15 breeds or fewer if less than 15 total
top_breeds_df_pie = df_counts_sorted.head(top_n_pie)
other_count_pie = df_counts_sorted.iloc[top_n_pie:]['Image Count'].sum()

labels_pie = top_breeds_df_pie['Breed Name'].tolist()
sizes_pie = top_breeds_df_pie['Image Count'].tolist()

if other_count_pie > 0: # Add 'Others' slice only if there are remaining breeds
    labels_pie.append('Others')
    sizes_pie.append(other_count_pie)

# Use a vibrant colormap
colors_pie = plt.colormaps.get_cmap('tab20c')(np.arange(len(labels_pie)))

plt.figure(figsize=(12, 12))
plt.pie(sizes_pie, labels=labels_pie, autopct='%1.1f%%', startangle=140, colors=colors_pie,
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.5},
        pctdistance=0.85)
plt.title(f'Top {top_n_pie} Dog Breeds by Image Count and Others')
plt.axis('equal')
plt.show()


# --- Visualization 3: Histogram of Image Counts Across Breeds ---
plt.figure(figsize=(10, 6))
# Using seaborn's histplot for potentially better aesthetics and options
sns.histplot(df_counts_sorted['Image Count'], bins=30, kde=True, edgecolor='black', color='lightgreen')
plt.xlabel('Number of Images per Breed')
plt.ylabel('Number of Breeds (Frequency)')
plt.title('Histogram of Image Counts Across Breeds')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# --- Visualization 4: Cumulative Distribution Function (CDF) Plot ---
plt.figure(figsize=(10, 6))
# Calculate CDF
counts_array = np.sort(df_counts_sorted['Image Count'])
y_cdf = np.arange(1, len(counts_array) + 1) / len(counts_array)
plt.plot(counts_array, y_cdf, marker='.', linestyle='-', color='purple')
plt.xlabel('Number of Images per Breed')
plt.ylabel('Cumulative Proportion of Breeds')
plt.title('Cumulative Distribution Function of Image Counts per Breed')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axvline(x=np.percentile(counts_array, 25), color='gray', linestyle=':', label='25th Percentile')
plt.axvline(x=np.percentile(counts_array, 50), color='red', linestyle='--', label='50th Percentile (Median)')
plt.axvline(x=np.percentile(counts_array, 75), color='gray', linestyle=':', label='75th Percentile')
plt.legend()
plt.show()


# --- Visualization 5: Top N Breeds - Horizontal Bar Chart (Focused View) ---
top_n_bar_focus = min(20, total_breeds) # Show top 20 breeds
plt.figure(figsize=(12, 10))
sns.barplot(x='Image Count', y='Breed Name', data=df_counts_sorted.head(top_n_bar_focus), palette='Blues_d')
plt.xlabel('Number of Images')
plt.ylabel('Dog Breed')
plt.title(f'Top {top_n_bar_focus} Dog Breeds by Image Count')
plt.tight_layout()
plt.show()


# --- Visualization 6: Bottom N Breeds - Horizontal Bar Chart (Focused View) ---
bottom_n_bar_focus = min(20, total_breeds) # Show bottom 20 breeds
plt.figure(figsize=(12, 10))
sns.barplot(x='Image Count', y='Breed Name', data=df_counts_sorted.tail(bottom_n_bar_focus), palette='Reds_d')
plt.xlabel('Number of Images')
plt.ylabel('Dog Breed')
plt.title(f'Bottom {bottom_n_bar_focus} Dog Breeds by Image Count (Potentially Underrepresented)')
plt.tight_layout()
plt.show()


print("\n--- All visualizations generated. Please review the plots and save the ones you find most suitable for your PPT. ---")