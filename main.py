#coding:utf-8
import filename_fix
import from_csv

if __name__ == '__main__':
    print('Please choose an operation\r\n'
              '[A] Download images from cfh according to the csv file.\r\n'
              '[B] Download images from gbif according to the csv file.\r\n'
              '[C] Update csv file with the number of images.\r\n'
              '[D] Train the model.\r\n'
              '[X] Exit.')
    while True:
        user_input = input('Press enter to end input: ')
        if user_input == 'A':
            from_csv.from_csv_in_cfh('butt.csv')
        elif user_input == 'B':
            from_csv.from_csv_in_gbif('butt.csv')
        elif user_input == 'C':
            from_csv.update_csv('butt.csv')
        elif user_input == 'D':
            print('not supported yet.')
        elif user_input == 'X':
            break
        else:
            print('Illegal input.')
