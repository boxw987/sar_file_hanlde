import csv
import re

def extract_data_to_csv(input_filepath, output_filepath):
    """
    Extracts data from the input file and saves it to a CSV file.

    The script looks for a block of data starting with a line that begins with 'all '
    (after stripping whitespace) and ends before a line that begins with 'Speed:'.
    """
    extracted_data = []
    capturing = False

    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            for line in infile:
                stripped_line = line.strip()
                if not stripped_line:  # Skip empty lines
                    continue

                # Modified condition: "all" must be followed by a space
                if stripped_line.startswith("all "):
                    capturing = True
                
                if stripped_line.startswith("Speed:"):
                    capturing = False
                    break  # Stop processing after this block

                if capturing:
                    # Split the line by one or more whitespace characters
                    parts = re.split(r'\s+', stripped_line)
                    if len(parts) == 7: # Ensure the line has the expected number of columns
                        extracted_data.append(parts)
                    # No special handling for 'all' line needed here if it's guaranteed to have 7 parts after split
                    # The re.split should handle the 'all ' line correctly if it conforms to the 7 column structure.

    except FileNotFoundError:
        print(f"错误：输入文件 '{input_filepath}' 未找到。")
        return
    except Exception as e:
        print(f"读取文件时发生错误： {e}")
        return

    if not extracted_data:
        print("未提取到数据。请检查输入文件格式和标记（'all ', 'Speed:'）。")
        return

    header = ['Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95']

    try:
        with open(output_filepath, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)
            writer.writerows(extracted_data)
        print(f"数据已成功提取并保存到 '{output_filepath}'")
    except Exception as e:
        print(f"写入 CSV 文件时发生错误： {e}")

if __name__ == '__main__':
    # Assuming 2.txt is in the same directory as the script
    # and the output CSV should also be in the same directory.
    input_file = '2.txt' 
    output_file = 'output_data.csv'
    
    # You can change the file paths here if needed:
    # input_file = r'e:\md_knowledge\python\tools\command_to_csv\2.txt'
    # output_file = r'e:\md_knowledge\python\tools\command_to_csv\output_data.csv'
    
    extract_data_to_csv(input_file, output_file)
