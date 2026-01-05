0. 首先，需先下載以下內容

	a. 若沒有conda請於以下網址安裝:https://repo.anaconda.com/miniconda/，下載版本為:Miniconda3-py310_25.7.0-2-Windows-x86_64.exe

	b. Yolo11m下載網址: https://docs.ultralytics.com/models/yolo11/#performance-metrics

	C. 大量街景照片網址: https://drive.google.com/file/d/1SKq4UCMMeJCW24NLslqYzdXQJ7_jfp1h/view?usp=drive_link


1. 請於終端機(ctrl+r，然後輸入CMD) 開啟輸入以下指令:

	a. 將CMD轉至您的檔案位置:cd /d C:\Users\user\Desktop\studio

	b. 建立虛擬環境於上述之位置: conda env create -f environment.yml --prefix .\.conda_env

2. 所有的程式碼我有以相對路徑做連結因此直接照以下分別運行即可:
	
	a. Relative Path Version.py ，該程式碼可針對筆者已收集好之相片與CSV檔進行辨識，並將辨識結果已文字生成新CSV

	b. Relative_Geojson.py 基於上個步驟生成的CSV生成Geojson檔進行連線

3.匯入QGIS:

	a. 匯CSV: 圖層(L) → 加入圖層 → 加入分隔文字圖層→選好檔案後→X欄位選lng，y欄位選lat→幾何圖形選TWD97→確認即可

	b. 匯json: 圖層(L) → 加入圖層 →加入向量圖層→選好檔案後確認即可


補充: 
	a.本檔案亦有保留已完成之現成CSV、Geojson、辨識後照片，同學亦可跳過0、1、2步驟直接進行3.

	b.goal_film是後續程式碼會將生成結果存於該位置，建議自行創建名稱為goal_film之空白資料夾。