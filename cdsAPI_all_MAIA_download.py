import cdsapi

variables = ['vertical_integral_of_northward_water_vapour_flux']#'2m_temperature', 'sea_ice_cover', 'vertical_integral_of_eastward_water_vapour_flux', 

ups = '1996'
years_all = [str(i) for i in range(1994, 2023)]
years_select = '2022'#[str(i) for i in range(2008,2023)]
#years_select.append(ups)

years1994_2003 = ['1994', '1995', '1996',
 '1997', '1998', '1999',
 '2000', '2001', '2002',
 '2003',
 ]
years2004_2013 = ['2004', '2005', '2006',
    '2007', '2008', '2009',
    '2010', '2011', '2012',
    '2013',
    ]
years2014_2022 = ['2014', '2015', '2016',
    '2017', '2018', '2019',
    '2020', '2021', '2022',
    ]


yearss = [years1994_2003, years2004_2013, years2014_2022]#[[str(i) for i in range(1994, 2004)]]#, [str(i) for i in range(2004, 2014)], [str(i) for i in range(2014, 2024)]]

path = '/Volumes/1TB-FREECOM/MAIA_data/'

for variable in variables:
    if variable == 'vertical_integral_of_northward_water_vapour_flux':
        years = years_select
    else:
        years = years_all

    for year in years:
        print('Downloading ' + variable + ' for year ' + year)
              
        fn = 'ERA5_download__' + variable + '_' + year + '.nc'
       
        c = cdsapi.Client()

        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': variable,
                'year': year,
                'month': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                ],
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'area': [
                    90, -180, 60,
                    180,
                ],
            },
            path + fn)