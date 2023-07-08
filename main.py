# coding:utf-8
import dataset_split
import file_fix
import from_csv
from dataset_augment import dataset_augment

if __name__ == '__main__':
    print('Please choose an operation\r\n'
          '[C]FH image download according to the csv file.\r\n'
          '[G]BIF image download according to the csv file.\r\n'
          '[U]pdate csv file with the number of images.\r\n'
          '[A]ugment data.\r\n'
          '[T]rain the model.\r\n'
          '[M]ove files according to the csv file.\r\n'
          '[F]ix file name illegal character.\r\n'
          '[R]emove empty files.\r\n'
          '[S]plit dataset.\r\n'
          'E[X]IT.')
    while True:
        user_input = input('Press enter to end input: ')
        if user_input == 'C':
            from_csv.from_csv_in_cfh('labels.csv')
        elif user_input == 'G':
            from_csv.from_csv_in_gbif('labels.csv')
        elif user_input == 'U':
            from_csv.update_csv('labels.csv', 'new_csv.csv')
        elif user_input == 'T':
            print('not supported yet.')
        elif user_input == 'X':
            break
        elif user_input == 'M':
            from_csv.from_csv_move_file('mv_files.csv')
        elif user_input == 'R':
            file_fix.empty_file()
        elif user_input == 'F':
            file_fix.filename()
        elif user_input == 'S':
            dataset_split.split()
        elif user_input == 'A':
            dataset_augment("/data/train/")
        else:
            print('Illegal input.')
