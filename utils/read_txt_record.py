# import pudb; pu.db
import sys
import numpy as np
import os
import xlwt
import matplotlib.pyplot as plt
import math
import xlrd
import pudb as pu

def read_one_record(record_path):
    # txt_path = r'C:\Users\Bowen Zhang\AppData\Local\Temp\fz3temp-2\gtsrb_record 9.txt'
    txt_path = record_path
    ini_c = 1e-1

    flags = []
    psnrs = []
    constants = []
    confs = []
    num_iters = []
    least_num_bss = []
    max_iter_psnr = []
    only_success_psnrs = []
    only_fail_psnrs = []
    linfs  =[]
    cost_time = []
    hamming_losses = []
    with open(txt_path) as f:
        for idx, line in enumerate(f.readlines()):
            if 'ave' in line:
                continue

            if 'uccess = True' in line:
                flags.append(1)
                # _psnr = line.split('psnr = ')[-1].split(',')[0]
                # if _psnr != 'nan':
                #     psnrs.append(np.abs(float(_psnr)))
            # if '\'constant\'' in line:
            #     # c = float(line.split('constant = ')[-1].split(',')[0])
            #     c = float(line.split('\'constant\': ')[-1].split(',')[0])
            #     constants.append(c)
            #     if c > ini_c:
            #         least_num_bs = np.round(np.log10(c/ini_c))+2
            #         least_num_bss.append(least_num_bs)
            #
            # if 'confidence' in line and not 'kwargs' in line:
            #     conf = float(line.split('confidence = ')[-1].rstrip('\n').split(',')[0])
            #     confs.append(conf)
            #
            # if 'L-inf' in line:
            #     linf = float(line.split('L-inf = ')[-1].split(',')[0])
            #     linfs.append(linf)
            elif 'uccess = False' in line:
                if 'hamming loss' in line:
                    _hloss = float(line.split('hamming loss = ')[1].split(',')[0])
                    if _hloss == 0:
                        flags.append(1)
                    else:
                        flags.append(0)
                else:
                    flags.append(0)
            else:
                pass

            if 'hamming loss' in line:
                hloss = float(line.split('hamming loss = ')[1].split(',')[0])

                if not math.isnan(hloss):
                    # import pudb; pu.db
                    hamming_losses.append(hloss)
                psnr = float(line.split('psnr = ')[-1])
                if not math.isnan(psnr):
                    # import pudb; pu.db
                    if hloss == 0:
                        only_success_psnrs.append(psnr)
                    else:
                        only_fail_psnrs.append(psnr)

                    psnrs.append(np.abs((psnr)))

            if 'lambda' in line:
                const = float(line.split('lambda = ')[1].split(',')[0])
                constants.append(const)

            # if '\'num_iteration\':' in line:
            #     num_iter = float(line.split('\'num_iteration\': ')[-1].split('}')[0])
            #     num_iters.append(num_iter)

            # if '\'num_iter\':' in line:
            #     num_iter = float(line.split('\'num_iter\': ')[-1].split(',')[0])
            #     num_iters.append(num_iter)

            # if 'num_iter' in line:
            if 'num_iter' in line:
                try: num_iter = float(line.split('num_iter =')[-1].split(',')[0])
                except ValueError: num_iter = 0
                if not math.isnan(num_iter):
                    num_iters.append(num_iter)
                else:
                    import pudb; pu.db
            if 'cost' in line and flags[-1] == 1:
                seconds = round(float(line.split('cost ')[1].split('seconds')[0]), 2)
                if not math.isnan(seconds):
                    # import pudb; pu.db
                    cost_time.append(seconds)


            # if 'constant' in line:
            #     c = float(line.split(', ')[0].split(': ')[-1])
            #     print('constant = ',c)
            #     constants.append(c)
            #     if c > ini_c:
            #         least_num_bs = np.round(np.log10(c/ini_c))+2
            #         least_num_bss.append(least_num_bs)
            # if 'confidence' in line:
            #     conf = float(line.split("'confidence': ")[1].rstrip('}\n'))
            #     print('conf = ',conf)
            #     confs.append(conf)
            # if 'num_iteration' in line:
            #     num_iter = float(line.split("'num_iteration': ")[1].rstrip('}\n').split(", 'conf'")[0])
            #     print('num_iter = ', num_iter)
            #     num_iters.append(num_iter)
            # if 'max_iter_psnr' in line:
            #     x = float(line.split("'max_iter_psnr': ")[1].rstrip('}\n'))
            #     max_iter_psnr.append(x)

    try:
        max_iter = np.max(num_iters)
    except:
        max_iter = np.nan
    print('Record path = \n', txt_path)
    ASR = np.sum(flags) / len(flags)
    print('ASR = {} / {} = {}\n'
          'mean psnr = {} / {} = {}\n'
          'mean only success psnr = {}\n'
          'mean only fail psnr  = {}\n'
          'mean constant = {} / {} = {}\n'
          'mean conf = {} / {} = {}\n'
          'mean num_iter = {}, max iter = {}\n'
          'mean least bs number = {}\n'
          'mean max iter psnr = {}\n'
          'mean l-inf = {}\n'
          'mean cost time = {} seconds' .format(
        np.sum(flags), len(flags), np.sum(flags) / len(flags),
        np.sum(psnrs), len(psnrs), np.mean(psnrs),
        np.mean(only_success_psnrs), np.mean(only_fail_psnrs),
        np.sum(constants), len(constants), np.mean(constants),
        np.sum(confs), len(confs), np.mean(confs),
        np.mean(num_iters), max_iter,
        np.mean(least_num_bss),
        np.mean(max_iter_psnr),
        np.mean(linfs),
        np.mean(cost_time)
    ))
    if math.isnan(np.mean(num_iters)):
        import pudb; pu.db
    return {'ASR': ASR,
            'AVE PSNR': np.mean(psnrs),
            'AVE ITER': np.mean(num_iters),
            'AVE TIME': np.mean(cost_time),
            'AVE LAMBDA': np.mean(constants),
            'AVE HAMMING LOSS': np.mean(hamming_losses)}


def set_style(name, height, bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style


def write_excel_xls(path, sheet_name, value):
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i][j])  # 像表格中写入数据（对应的行和列）
    workbook.save(path)  # 保存工作簿
    # print("xls格式表格写入数据成功！")


def write_excel_xls_append(path, value):
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlrd.open_workbook(path)  # 打开工作簿
    sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
    worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
    rows_old = worksheet.nrows  # 获取表格中已存在的数据的行数
    new_workbook = xlrd.copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
    new_worksheet = new_workbook.get_sheet(0)  # 获取转化后工作簿中的第一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            new_worksheet.write(i+rows_old, j, value[i][j])  # 追加写入数据，注意是从i+rows_old行开始写入
    new_workbook.save(path)  # 保存工作簿
    # print("xls格式表格【追加】写入数据成功！")


if __name__ == '__main__':

    MODEL_ROOT = r'G:\bowen'

    # txt_path = '/media/hdddati2/bzhang/trained_models/gtsrb/ECOC/adversarials/CWB/C0I1e-4LR1e-3NBS5MI0.1k-output_decoded/gtsrb_record.txt'
    folder_path = os.path.join(MODEL_ROOT, 'VOC12/InceptionV3-low_performance/adversarials/LOTS/HMloss/RealError/200imgs')
    records = {}
    setting_sets = []

    stps = []
    confs = []
    mxis = []
    for folder in os.listdir(folder_path):
        if not os.path.isdir(os.path.join(folder_path, folder)):
            continue
        mxi = int(float(folder.split('MI')[1].split('Conf')[0]))
        Conf = int(float(folder.split('Conf')[1]))
        stp = round(float(folder.split('STP=')[1].split('MI')[0]), 2)

        stps.append(stp)
        confs.append(Conf)
        mxis.append(mxi)

        setting_sets.append((stp, Conf))

        txt_path = os.path.join(folder_path, folder, 'record.txt')
        results = read_one_record(txt_path)
        asr = results['ASR']
        psnr = results['AVE PSNR']
        aveiter = results['AVE ITER']
        cost_time = results['AVE TIME']
        const_lambda = results['AVE LAMBDA']
        hamming_loss = results['AVE HAMMING LOSS']
        # plt.figure()
        # plt.hist(num_iters)
        # plt.axvline(np.mean(num_iters), color = 'red', linewidth = 2)
        # min_ylim, max_ylim = plt.ylim()
        # plt.text(np.mean(num_iters) * 1.1, max_ylim * 0.9, 'Mean: {:.2f}'.format(np.mean(num_iters)))
        # plt.savefig(os.path.join(folder_path, folder,'STP{}MI{}IterDistribute.png'.format(stp, Conf)))

        records.update({'STP{}Conf{}MXI{}'.format(stp, Conf, mxi): (round(float(asr), 4), round(float(psnr), 2), round(float(hamming_loss), 2), round(float(aveiter), 2))})

    stps = list(set(stps))
    stps.sort()
    confs = list(set(confs))
    confs.sort()
    mxis = list(set(mxis))
    mxis.sort()

    # print('before sort')
    # for idx, (key, value) in enumerate(records.items()):
    #     if idx > 3:
    #         break
    #     print(key, value, np.prod(value))
    # # b = sorted(records.items(), key=lambda item: np.prod(item[1][:2]), reverse=True)
    # b = sorted(records.items(), key=lambda item: np.prod(item[1][2]), reverse=True)
    # print('after sort')
    # for idx, v in enumerate(b):
    #     if idx>10:
    #         break
    #     print(v)


    f = xlwt.Workbook()
    sheet1 = f.add_sheet('grid search', cell_overwrite_ok=True)
    # stps = np.concatenate((np.linspace(0.01,0.04,4), np.linspace(0.05, 1, 20)))
    # mxis = np.linspace(20,200,10)
    # stps = np.linspace(0.05, 0.1, 20)
    # stps = np.concatenate((np.linspace(0.01,0.04,4),np.linspace(0.05, 1, 20)))
    # mxis = [200]

    # stps = np.concatenate((np.linspace(0.01,0.04,4),np.linspace(0.05, 1, 20)))
    # confs = [0,1,3,5,10,30]

    row0 = confs
    colum0 = stps
    # 写第一行
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))
    # 写第一列
    for i in range(0, len(colum0)):
        sheet1.write(i + 1, 0, colum0[i], set_style('Times New Roman', 220, True))

    for r, rvalue in enumerate(colum0):
        for c, cvalue in enumerate(row0):
            try:
                # sheet1.write(r + 1, c + 1, str(records['STP{}Conf{}MXI{}'.format(round(stp,2), 0, int(mxi))]))
                sheet1.write(r + 1, c + 1, str(records['STP{}Conf{}MXI{}'.format(round(rvalue, 2), int(cvalue), mxis[0])]))
            except KeyError:
                pass

    # sheet1.write(1, 3, '2006/12/12')
    # sheet1.write_merge(6, 6, 1, 3, '未知')  # 合并行单元格
    # sheet1.write_merge(1, 2, 3, 3, '打游戏')  # 合并列单元格
    # sheet1.write_merge(4, 5, 3, 3, '打篮球')

    f.save(os.path.join(folder_path, 'resultstest.xls'))
