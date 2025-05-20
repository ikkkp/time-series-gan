"""
@File  :local.py
@Author:Ezra Zephyr
@Date  :2025/5/210:34
@Desc  :
"""
import platform, subprocess, json

def get_system_info():
    info = platform.uname()
    return {
        '操作系统': f"{info.system} {info.release}",
        '计算机名称': info.node,
        '处理器': info.processor,
    }

def get_memory_info():
    out = subprocess.check_output('systeminfo', shell=True, text=True)
    for line in out.splitlines():
        if '总物理内存' in line:
            return {'总物理内存': line.split(':')[-1].strip()}
    return {}

def get_gpu_info():
    try:
        out = subprocess.check_output(
            'wmic path win32_VideoController get Name,AdapterRAM,DriverVersion /format:csv',
            shell=True, text=True
        )
        gpus = []
        for row in out.splitlines()[2:]:
            if row:
                _, name, ram, drv = [c.strip() for c in row.split(',')]
                gpus.append({'名称': name, '显存(Bytes)': ram, '驱动版本': drv})
        return gpus
    except subprocess.CalledProcessError:
        # 如果在 CMD 中失败，可在 PowerShell 中执行：
        # Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion
        return ['请在 PowerShell 中执行：Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion']

if __name__ == '__main__':
    res = {**get_system_info(), **get_memory_info(), 'GPU 信息': get_gpu_info()}
    print(json.dumps(res, ensure_ascii=False, indent=2))
