import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

def split_metadata(metadata_filename):
    # Read the CSV file into a DataFrame
    df_meta = pd.read_csv(metadata_filename)
    
    # Group the DataFrame by the 'language' column
    grouped = df_meta.groupby('language')

    # Initialize empty DataFrames for training, validation, and testing
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()

    # Iterate over each group
    for _, group in grouped:
        # Split each group into training, validation, and testing sets
        train_group, temp_group = train_test_split(group, test_size=0.3, random_state=42)
        val_group, test_group = train_test_split(temp_group, test_size=0.67, random_state=42)
        
        # Append the splits to the respective DataFrames
        train_df = pd.concat([train_df, train_group])
        val_df = pd.concat([val_df, val_group])
        test_df = pd.concat([test_df, test_group])

    return train_df, val_df, test_df

def copy_images(train_df, val_df, test_df, source_folder, destination_folder):
   
    # Create destination folders if they don't exist
    train_dest_folder = os.path.join(destination_folder, 'training')
    val_dest_folder = os.path.join(destination_folder, 'validation')
    test_dest_folder = os.path.join(destination_folder, 'testing')
    os.makedirs(train_dest_folder, exist_ok=True)
    os.makedirs(val_dest_folder, exist_ok=True)
    os.makedirs(test_dest_folder, exist_ok=True)

   # Copy images for training set
    for index, row in train_df.iterrows():
        source_path = os.path.join(source_folder, row['id'] + '.jpg')
        train_dest_path = os.path.join(train_dest_folder, row['id'] + '.jpg')
        shutil.copyfile(source_path, train_dest_path)

    # Copy images for validation set
    for index, row in val_df.iterrows():
        source_path = os.path.join(source_folder, row['id'] + '.jpg')
        val_dest_path = os.path.join(val_dest_folder, row['id'] + '.jpg')
        shutil.copyfile(source_path, val_dest_path)

    # Copy images for testing set
    for index, row in test_df.iterrows():
        source_path = os.path.join(source_folder, row['id'] + '.jpg')
        test_dest_path = os.path.join(test_dest_folder, row['id'] + '.jpg')
        shutil.copyfile(source_path, test_dest_path)

def create_char_mapping(charlist_file):
    char_mapping = {}
    with open(charlist_file, 'r') as file:
        for index, char in enumerate(file):
            char = char.strip()  # Remove leading/trailing whitespaces
            if char == "<SPACE>":
                char = " "  # Replace "<space>" with actual space character
            char_mapping[char] = str(index)
    return char_mapping

def convert_text(text, char_mapping):
    converted_text = ""

    for char in text:
        if char in ['\t', '\n', '\xa0']:
            char = " "
    
        if char in char_mapping:
            converted_text += char_mapping[char] + " "  # Add a space after each converted character
        else:
            print("Didn't find character " + char)
            converted_text += char  # Keep unchanged if not found in mapping

    return converted_text.strip()

def create_labels_folder(dataframe, folder_path, char_mapping):

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Iterate through the DataFrame
    for index, row in dataframe.iterrows():
        file_name = str(row['id']) + ".tru"  # Constructing the file name from 'id'
        text_content = row['text']  # Extracting text content from 'text' column
        
        # Convert text_content using char_mapping
        converted_text_content = convert_text(text_content, char_mapping)

        # Constructing the full file path
        file_path = os.path.join(folder_path, file_name)
        
        # Writing the text content to the file
        with open(file_path, "w") as file:
            file.write(converted_text_content)

if __name__ == "__main__":
    # Specify paths
    metadata_filename = 'line_meta.csv'
    source_folder = 'lines'  # Path to the folder containing all images
    destination_folder = 'train_val_test_split'  # Folder where split images will be saved
    char_mapping = create_char_mapping("./CNN/samples/CHAR_LIST")
    
    # Split the metadata
    train_df, val_df, test_df = split_metadata(metadata_filename)

    # Copy images into separate folders
    copy_images(train_df, val_df, test_df, source_folder, destination_folder)

    # Save DataFrames to CSV files in the current directory
    train_df.to_csv('train_metadata.csv', index=False)
    val_df.to_csv('val_metadata.csv', index=False)
    test_df.to_csv('test_metadata.csv', index=False)

    # Save a folder for labels in each folder
    create_labels_folder(train_df, './train_val_test_split/training/labels', char_mapping)
    create_labels_folder(val_df, './train_val_test_split/validation/labels', char_mapping)
    create_labels_folder(test_df, './train_val_test_split/testing/labels', char_mapping)

    # Save the list of IDs from train_df and test_df as a text file
    train_images_list = train_df['id'].tolist()
    with open('train_images_list.txt', 'w') as f:
        for item in train_images_list:
            f.write("%s\n" % item)

    val_images_list = val_df['id'].tolist()
    with open('val_images_list.txt', 'w') as f:
        for item in val_images_list:
            f.write("%s\n" % item)

    test_images_list = test_df['id'].tolist()
    with open('test_images_list.txt', 'w') as f:
        for item in test_images_list:
            f.write("%s\n" % item)

    # Print the shape of training, validation, and testing sets
    print("Training DataFrame shape:", train_df.shape)
    print("Validation DataFrame shape:", val_df.shape)
    print("Testing DataFrame shape:", test_df.shape)
    
    items = os.listdir('train_val_test_split/testing')
    num_items = len(items)
    
    print(f"There are {num_items} items in the testing folder.")

    items = os.listdir('train_val_test_split/validation')
    num_items = len(items)
    
    print(f"There are {num_items} items in the validating folder.")

    items = os.listdir('train_val_test_split/training')
    num_items = len(items)
    
    print(f"There are {num_items} items in the training folder.")