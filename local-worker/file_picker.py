import subprocess
import sys

def open_windows_file_dialog():
    # 1. Ưu tiên Tkinter (Bật cửa sổ chọn file Windows tức thì trong 0.01s và nổi lên trên cùng)
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        root.focus_force()
        file_path = filedialog.askopenfilename(
            title="Chọn video Douyin từ máy tính",
            filetypes=[
                ("Video Files", "*.mp4 *.mov *.mkv *.avi *.webm *.flv"),
                ("All Files", "*.*")
            ]
        )
        root.destroy()
        if file_path:
            return file_path
    except Exception:
        pass

    # 2. Dự phòng bằng PowerShell Windows Forms nếu không có Tkinter
    ps_script = """
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = "Chọn video từ máy tính của bạn"
$dialog.Filter = "Video Files (*.mp4;*.mov;*.mkv;*.avi;*.webm)|*.mp4;*.mov;*.mkv;*.avi;*.webm|All Files (*.*)|*.*"
$dialog.InitialDirectory = [Environment]::GetFolderPath("UserProfile") + "\\Downloads"
$res = $dialog.ShowDialog()
if ($res -eq [System.Windows.Forms.DialogResult]::OK) {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    Write-Output $dialog.FileName
}
"""
    res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], capture_output=True, text=True, encoding="utf-8")
    return res.stdout.strip()

if __name__ == "__main__":
    pass
