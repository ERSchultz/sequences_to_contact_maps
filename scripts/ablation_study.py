import csv


def main():
    key = {427: 'original',
            430: r'predict $S$ (instead of $S^\dag$)',
            431: 'without H bonded',
            432: 'without ContactDistance',
            433: 'without meanConstact Distance',
            434: 'without genetic distance norm',
            435: 'without signconv (eigenvectors replaced with constant)',
            436: 'with gatv2conv instead of modified',
            437: 'with mean y_norm instead of mean_fill',
            438: 'without signconv (eigenvectors are naively included)',
            }

    

if __name__ == '__main__':
    main()
