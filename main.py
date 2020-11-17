#coding:utf-8
import filename_fix
import from_csv

if __name__ == '__main__':
    print('Please choose an operation\r\n'
          '[C]FH image download according to the csv file.\r\n'
          '[G]BIF image download according to the csv file.\r\n'
          '[U]pdate csv file with the number of images.\r\n'
          '[T]rain the model.\r\n'
          '[M]ove files according to the csv file.\r\n'
          'E[X]IT.')
    while True:
        user_input = input('Press enter to end input: ')
        if user_input == 'C':
            from_csv.from_csv_in_cfh('butt.csv')
        elif user_input == 'G':
            from_csv.from_csv_in_gbif('butt.csv')
        elif user_input == 'U':
            from_csv.update_csv('butt.csv', 'new_csv.csv')
        elif user_input == 'T':
            print('not supported yet.')
        elif user_input == 'X':
            break
        elif user_input == 'M':
            from_csv.from_csv_move_file('mv_files.csv')
        else:
            print('Illegal input.')
