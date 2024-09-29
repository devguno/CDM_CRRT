import numpy as np
import pandas as pd
import glob
import tarfile
import matplotlib.pyplot as plt
import os

import glob

# 경로 수정
path_base = "PA020138\\2022\\"
list_file = glob.glob(path_base + '*.LOX')

# 파일 개수 출력
len(list_file)

extmap = {
    "pci": ("Network config", ["ascii", "ini", "split", "strip"]),
    "pca": None,  # Some binary data, skip it
    "pcu": ("Therapy config", ["ascii", "ini", "split", "strip"]),
    "pcm": ("Machine config", ["ascii", "split"]),
    "plr": ("System events", ["ascii", "split", "strip", "noemptylines"]),
    "ple": ("User events", ["utf-16", "split", "strip", "csv"]),
    "plp": ("Pressure", ["utf-8", "split", "strip", "csv"]),
    "pls": ("Fluids", ["ascii", "split", "strip", "csv", ]),
    "ply": ("Syringe", ["ascii", "split", "strip", "csv", ]),
    "plc": ("PLC", ["ascii", "split", "strip", "csv", ]),
    "plt": ("Tare", ["ascii", "split", "strip", "csv", ]),
    "pli": ("PLI", ["ascii", "split", "strip", "csv", ]),
    "pll": ("PLL", ["ascii", "split", "strip", "csv", ])
}

def get_loxfile_data(fname):
    """
    Returns all the data contained in the loxfile
    :param fname:
    :return: dictionary
    """

    ret = {}

    tar = tarfile.open(fname, "r:gz")

    for member in tar.getnames():
        _ign, ext = map(str.lower, member.split("."))

        f = tar.extractfile(member)
        if ext in extmap:
            if extmap[ext] is None:
                continue

            desc, extra = extmap[ext]
            data = f.read()

            for elem in extra:
                if elem == "strip":
                    data = [x.strip() for x in data]
                    continue

                if elem in ["utf-8", "utf-16", "ascii"]:
                    data = data.decode(elem)
                    continue

                if elem == "split":
                    data = data.split("\n")
                    continue

                if elem == "csv":
                    # data = [x for x in csv.reader(data, delimiter=';')]
                    data = [x.split(';') for x in data]
                    continue

                if elem == "noemptylines":
                    data = [x for x in data if x]
                    continue

            ret[desc] = data

        else:
            print('Unknown extension : {}'.format(ext))

    return ret

dict_file = get_loxfile_data(list_file[0])
dict_file.keys()

dict_file['Network config'] # IP 관련 정보
dict_file['Therapy config'] # 뭔자 처치와 관련된 것 같은데 뭔지 모르겠음
dict_file['Machine config'] # 기계 설정같은데 뭔지 잘 모르겠음
dict_file['System events'] # 뭔가 event인데 뭔지 잘 모르겠음
dict_file['Syringe'] # 별 정보 없음
dict_file['Tare'][6] # bag 바꿀 때 무게가 기록된 것

dict_file['PLC'][6] # Scale deviation?
dict_file['PLI'][6] # ?
dict_file['PLL'][8] # ?

dict_file['User events'][26] # <<< 투석중 발생한 event

try:
    table_event = pd.DataFrame(dict_file['User events'][27:], columns=dict_file['User events'][26])
except:
    table_event = pd.DataFrame(dict_file['User events'][27:], columns=dict_file['User events'][26]+['None'])
table_event = table_event.iloc[:,1:]

table_fluid = pd.DataFrame(dict_file['Fluids'][7:], columns=dict_file['Fluids'][6])
table_fluid = table_fluid.iloc[:,1:]
table_pressure = pd.DataFrame(dict_file['Pressure'][7:], columns=dict_file['Pressure'][6])
table_pressure = table_pressure.iloc[:,1:]
table_pressure


### Event 관련
# (0) 투석 시작 metadata
# 416: 환자 인식 번호
# 550: 요법 종류
# 17: 혈액 속도 (mL/min)
# 22: 사전 혈액 펌프 (mL/hr)
# 20: 대체용액 (mL/hr)
# 21: 투석액 (mL/hr)
# 24: 환자 수분 제거 (mL/hr)

# (1) 투석 시작
# 16: 치료가 시작되었습니다(실행 모드).
# 20: 재시작을 선택했습니다.

# (2) 투석 alarm
# 5: 경고: 필터 응고됨

# (3) 투석 끝
# 19: 중지를 선택했습니다.
# 21: 치료 종료를 선택했습니다.

list_type = ['416', '550', '17', '22', '20', '21', '24', '16', '20', '279', '5', '19', '21']
list_type_t = ['환자 인식 번호:', '요법 종류:', '혈액', '사전 혈액 펌프', '대체용액', '투석액',
               '환자 수분 제거', '치료가 시작되었습니다(실행 모드).', '재시작을 선택했습니다.',
               '보고: 필터 응고가 진행중', '경고: 필터 응고됨', '중지를 선택했습니다.', '치료 종료를 선택했습니다.']
list_type_cod = ['PT_ID','CRRT_type','BFR','Pre','Replace','Dialysate','UF',
                 'HD_start','HD_restart', 'Warning_coag', 'Filter_coag','HD_suspend','HD_end']


### 모든 Type 종류 뽑기 ###
for i, t_file in enumerate(list_file):
    print(i)
    dict_file = get_loxfile_data(t_file)
    try:
        table_event = pd.DataFrame(dict_file['User events'][27:], columns=dict_file['User events'][26])
    except:
        table_event = pd.DataFrame(dict_file['User events'][27:], columns=dict_file['User events'][26]+['None'])
    table_event = table_event.iloc[:,1:]
    col_type = table_event[['Type(cod)', 'Type']]
    if i == 0:
        all_col_type = col_type
    else:
        all_col_type = pd.concat([all_col_type, col_type], axis=0)

all_col_type.shape
all_col_type.value_counts()
# all_col_type.value_counts().to_excel('type.xlsx')


### 특정 기계의 모든 event 뽑기 ###
for n, t_file in enumerate(list_file):
    print(n)
    machine_name = t_file.split('\\')[-3]
    dict_file = get_loxfile_data(t_file)

    try:
        table_event = pd.DataFrame(dict_file['User events'][27:], columns=dict_file['User events'][26])
    except:
        table_event = pd.DataFrame(dict_file['User events'][27:], columns=dict_file['User events'][26]+['None'])
    table_event = table_event.iloc[:,1:]

    # 초 단위 정보는 삭제함
    table_event = table_event[table_event['Time'] != ''] # 가끔 blank slot이 있는 듯
    table_event['Time'] = table_event['Time'].str[:-2] + '00'
    table_event['Time'] = pd.to_datetime(table_event['Time'])
    table_event.sort_values(by='Time', inplace=True)
    table_event.reset_index(drop=True, inplace=True)

    for i in range(len(list_type)):
        curr_type = list_type[i]
        curr_type_t = list_type_t[i]
        curr_type_name = list_type_cod[i]
        col_sub = table_event[(table_event['Type(cod)'] == curr_type) & (table_event['Type'] == curr_type_t)][['Time', 'Sample']]
        col_sub['Sample'] = np.where(col_sub['Sample'] == '', 'O', col_sub['Sample'])
        col_sub.rename(columns={'Sample':curr_type_name}, inplace=True)
        if i == 0:
            all_col = col_sub
        else:
            all_col = pd.merge(all_col, col_sub, on='Time', how='outer')

    all_col.sort_values(by='Time', inplace=True)
    all_col['Machine'] = machine_name
    # all_col['file_idx'] = n

    if n == 0:
        merged_event = all_col
    else:
        merged_event = pd.concat([merged_event, all_col], axis=0)

merged_event.sort_values(by=['Machine', 'Time'], inplace=True)
merged_event.drop_duplicates(inplace=True)
merged_event.reset_index(drop=True, inplace=True)
merged_event


### 각 event를 session 별로 분리함
# 분리 기준 => HD_start ~ HD_end가 찍혀있는 기준으로
merged_event['Sess'] = 0
list_machine = merged_event['Machine'].drop_duplicates()
sess_num = 1

for n, machine in enumerate(list_machine):
    machine_event = merged_event[merged_event['Machine'] == machine].copy()

    start_array = (~machine_event['PT_ID'].isna()) &  (machine_event['HD_start'] == 'O')
    end_array = ~machine_event['HD_end'].isna()
    idx_array = np.zeros(machine_event.shape[0])
    find_start = True
    find_pos = 0
    find_end = machine_event.shape[0]
    curr_start = None
    curr_end = None
    complete = 0

    while find_pos != find_end:
        print('screening ... machine {} / {}, position {} / {}'.format(n, len(list_machine), find_pos, find_end))
        if find_start == True:
            checker = start_array.iloc[find_pos]
            if checker == True:
                if complete == 1:
                    idx_array[curr_start:curr_end+1] = sess_num
                    complete = 0
                    sess_num += 1
                curr_start = find_pos
                find_start = False
                # find_pos += 1
                # curr_start가 찍힌 곳에 curr_end도 찍힐 수 있기 때문에 find_pos +=1이 없음
            else:
                # start를 찾고 있는데 end가 한 번 더 확인된다면 end를 확장함
                checker = end_array.iloc[find_pos]
                if checker == True:
                    curr_end = find_pos
                find_pos += 1
        else:
            checker = end_array.iloc[find_pos]
            if checker == True:
                curr_end = find_pos
                complete = 1
                find_start = True
                find_pos += 1
            else:
                find_pos += 1

    if complete == 1:
        idx_array[curr_start:curr_end+1] = sess_num
        complete = 0
        sess_num += 1

    merged_event.loc[merged_event['Machine'] == machine,'Sess'] = idx_array

# merged_event.to_csv('merged_table.csv', index=False, encoding='ANSI')


### session에 대한 quality control
# Start, End만 사용
merged_event_valid = merged_event[merged_event['Sess'] != 0].reset_index(drop=True)

# ID 중 이상한 문자들이 있는 경우가 있음
# 이것들은 최초 ID 입력 후 바로 다음 ROW에 생성되며 null data이므로 모두 삭제함
merged_event_valid['PT_ID'].drop_duplicates().tolist()

merged_event_valid['Invalid'] = np.where(merged_event_valid['PT_ID'].isna(), 0,
                                np.where(merged_event_valid['PT_ID'].str.len() == 20, 0, 1))
merged_event_valid = merged_event_valid[merged_event_valid['Invalid'] != 1]
merged_event_valid = merged_event_valid.drop('Invalid', axis=1).reset_index(drop=True)

# 파일을 저장할 디렉터리 경로
save_path = path_base + 'Analysis\\'
# 경로가 존재하지 않으면 생성
if not os.path.exists(save_path):
    os.makedirs(save_path)

merged_event_valid.to_csv(path_base + 'Analysis\\merged_table_valid.csv', index=False)

merged_event_valid.columns


### 특정 기계의 모든 Pressure 및 Fluid 뽑기 ###
for n, t_file in enumerate(list_file):
    print(n)
    machine_name = t_file.split('\\')[-3]
    dict_file = get_loxfile_data(t_file)

    table_fluid = pd.DataFrame(dict_file['Fluids'][7:], columns=dict_file['Fluids'][6])
    table_fluid = table_fluid.iloc[:,1:]
    table_fluid['Time'] = pd.to_datetime(table_fluid['Time'])
    table_fluid.sort_values(by='Time', inplace=True)

    table_pressure = pd.DataFrame(dict_file['Pressure'][7:], columns=dict_file['Pressure'][6])
    table_pressure = table_pressure.iloc[:,1:]
    table_pressure['Time'] = pd.to_datetime(table_pressure['Time'])
    table_pressure.sort_values(by='Time', inplace=True)

    table_metadata = pd.merge(table_fluid, table_pressure, on='Time', how='outer')
    table_metadata['Machine'] = machine_name
    # table_metadata['file_idx'] = n

    if n == 0:
        merged_metadata = table_metadata
    else:
        merged_metadata = pd.concat([merged_metadata, table_metadata], axis=0)

merged_metadata.sort_values(by=['Machine', 'Time'], inplace=True)
merged_metadata.drop_duplicates(inplace=True)
merged_metadata.reset_index(drop=True, inplace=True)


### event의 sess_num과 matching
merged_metadata['Sess'] = 0
array_sess = np.arange(sess_num)[1:]
for sess in array_sess:
    print(sess)
    curr_hd = merged_event_valid[merged_event_valid['Sess'] == sess]
    t_start = curr_hd['Time'].iloc[0]
    t_end = curr_hd['Time'].iloc[-1]
    name_machine = curr_hd['Machine'].iloc[0]
    array_valid = (merged_metadata['Time'] >= t_start) & (merged_metadata['Time'] <= t_end) & (merged_metadata['Machine'] == name_machine)
    merged_metadata.loc[array_valid, 'Sess'] = sess

merged_metadata.to_csv(path_base + 'Analysis\\merged_metadata.csv', index=False, encoding='utf-8')
merged_metadata.columns


### Exploration
merged_event_valid
merged_metadata

merged_event_valid['Sess'].drop_duplicates()
merged_event_valid['PT_ID'].dropna().drop_duplicates() # 79명의 환자
merged_event_valid['Time'].min() # 2022-01-05
merged_event_valid['Time'].max() # 2022-07-08
merged_event_valid['CRRT_type'].dropna().drop_duplicates() # CVVHDF, HP 2종류

# 환자별/세션별 투석 횟수 및 시간
time_table = merged_event_valid[['PT_ID', 'Time', 'Sess']]
time_table = time_table.fillna(method='ffill')
time_table = time_table.sort_values(by=['PT_ID', 'Time', 'Sess'], ascending=[True, True, True]).reset_index(drop=True)

time_table_per_ID = time_table.drop_duplicates(subset=['PT_ID']).reset_index(drop=True)
time_table_rev_per_ID = time_table.sort_values(by=['PT_ID', 'Time'], ascending=[True, False]).drop_duplicates(subset=['PT_ID']).reset_index(drop=True)
time_table_per_ID['Time_end'] = time_table_rev_per_ID['Time']
time_table_per_ID['Sess_end'] = time_table_rev_per_ID['Sess']

time_table_per_Sess = time_table.drop_duplicates(subset=['PT_ID', 'Sess']).reset_index(drop=True)
time_table_rev_per_Sess = time_table.sort_values(by=['PT_ID', 'Sess', 'Time'], ascending=[True, True, False]).drop_duplicates(subset=['PT_ID', 'Sess']).reset_index(drop=True)
time_table_per_Sess['Time_end'] = time_table_rev_per_Sess['Time']
time_table_per_Sess['Elapsed'] = (time_table_per_Sess['Time_end'] - time_table_per_Sess['Time']) / np.timedelta64(1, 'h')

time_table_per_ID # 크게 의미는 없는 것 같은데

time_table_per_Sess # 567개의 session이 존재함
time_table_per_Sess['Elapsed'].mean() # 한 번 투석 당 15.7시간 정도 유지됨
time_table_per_Sess.groupby('PT_ID')['Elapsed'].sum().mean() # 한 환자 당 112시간 정도 투석을 진행함


### Session statistics
table_summary = pd.DataFrame(columns=['Sess', 'Machine_ID', 'Pt_ID', 'Start_time', 'End_time', 'Event_Duration', 'Metadata_Duration', 'Warning', 'Coagulation'])

for i, sess in enumerate(array_sess):
    matched_event = merged_event_valid[merged_event_valid['Sess'] == sess]
    matched_metadata = merged_metadata[merged_metadata['Sess'] == sess]

    machine_id = matched_event['Machine'].iloc[0]
    pt_id = matched_event['PT_ID'].iloc[0]
    start_time_event = matched_event['Time'].iloc[0]
    end_time_event = matched_event['Time'].iloc[-1]
    event_duration = (end_time_event - start_time_event) / np.timedelta64(1, 'm')

    if matched_metadata.shape[0] != 0:
        start_time_metadata = matched_metadata['Time'].iloc[0]
        end_time_metadata = matched_metadata['Time'].iloc[-1]
        metadata_duration = (end_time_metadata - start_time_metadata) / np.timedelta64(1, 'm')
    else:
        metadata_duration = np.nan

    t_warning = matched_event[matched_event['Warning_coag'] == '(o)']
    if t_warning.shape[0] != 0:
        time_warning = t_warning['Time'].iloc[0]
    else:
        time_warning = np.nan

    t_coagulation = matched_event[matched_event['Filter_coag'] == '(o)']
    if t_coagulation.shape[0] != 0:
        time_coagulation = t_coagulation['Time'].iloc[0]
    else:
        time_coagulation = np.nan

    table_summary.loc[i,:] = [sess, machine_id, pt_id, start_time_event, end_time_event, event_duration, metadata_duration, time_warning, time_coagulation]

table_summary.head()
table_summary.to_csv('table_summary.csv', index=False, encoding='utf-8')

table_valid = table_summary[(table_summary['Metadata_Duration'] >= 60) & (table_summary['Event_Duration']*0.9 < table_summary['Metadata_Duration'])]

(table_valid['Event_Duration']/60).mean()
(table_valid['Event_Duration']/60).median()

plt.hist(table_valid['Event_Duration']/60, bins=15)
plt.show()

table_valid['Time_forecast'] = (table_valid['Coagulation']-table_valid['Warning'])/np.timedelta64(1, 'm')

TP = table_valid[~table_valid['Warning'].isna() & ~table_valid['Coagulation'].isna()].shape[0]
FP = table_valid[~table_valid['Warning'].isna() & table_valid['Coagulation'].isna()].shape[0]
FN = table_valid[table_valid['Warning'].isna() & ~table_valid['Coagulation'].isna()].shape[0]
TN = table_valid[table_valid['Warning'].isna() & table_valid['Coagulation'].isna()].shape[0]

sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)
precision = TP/(TP+FP)
f1 = 2*sensitivity*precision/(sensitivity+precision)

table_valid['Time_forecast'].dropna().shape[0]
table_fast = table_valid[table_valid['Time_forecast'] < 120]
table_ultrafast = table_valid[table_valid['Time_forecast'] <= 5]
table_ultrafast['Time_forecast'].shape[0]

plt.subplot(1,2,1)
plt.hist(table_valid['Time_forecast'], bins=60)
plt.xlabel('minutes')
plt.subplot(1,2,2)
plt.hist(table_fast['Time_forecast'], bins=60)
plt.xlabel('minutes')
plt.show()


table_valid['Pre_coag'] = (table_valid['Coagulation']-table_valid['Start_time'])/np.timedelta64(1, 'm')
table_valid['Post_coag'] = (table_valid['End_time']-table_valid['Coagulation'])/np.timedelta64(1, 'm')

pre_coag = table_valid['Pre_coag'].dropna()
post_coag = table_valid['Post_coag'].dropna()

(pre_coag/60).mean()
(pre_coag/60).median()

(post_coag).mean()
(post_coag).median()

plt.subplot(1,2,1)
plt.hist(pre_coag/60, bins=15)
plt.title('Elapsed time from CRRT start to coagulation')
plt.xlabel('hours')
plt.subplot(1,2,2)  
plt.title('Elapsed time from coagulation to CRRT end')
plt.hist(post_coag, bins=15)
plt.xlabel('minutes')
plt.show()

table_valid.to_csv('table_valid.csv', index=False, encoding='utf-8')

