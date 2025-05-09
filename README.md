# Laporan Proyek Pertama Machine Learning Expert Dicoding: Predictive Analytics - Diabet (split diabet n non) - Sarah Adibah

## Domain Proyek

*Domain proyek ini membahas permasalahan dalam bidang kesehatan, khususnya mengenai prediksi status diabetes berdasarkan data klinis dan demografis pasien.*
 
Proyek ini bertujuan untuk memprediksi apakah seseorang menderita diabetes atau tidak dengan menggunakan fitur-fitur seperti usia, riwayat merokok, tekanan darah, indeks massa tubuh (_BMI_), kadar HbA1c, dan kadar glukosa darah. Data yang digunakan mencakup beragam ras, lokasi, dan kondisi kesehatan pasien, sehingga dapat digunakan untuk membangun model klasifikasi yang dapat membantu deteksi dini diabetes secara lebih akurat.

<img src="https://cdn.rri.co.id/infografis/images/1676072537-prevalensi_diabetes_di_indonesia.jpg" alt="Data Diabetes" title="Data Diabetes" width="100%">

Diabetes merupakan salah satu penyakit kronis yang menjadi perhatian global karena jumlah penderitanya yang terus meningkat dari tahun ke tahun. Data dari International Diabetes Federation (IDF) menunjukkan bahwa pada tahun 2021, terdapat sekitar 537 juta penderita diabetes di seluruh dunia. Angka ini diproyeksikan akan meningkat menjadi 643 juta pada tahun 2030 dan mencapai 783 juta pada tahun 2045. [[1]](http://sehatnegeriku.kemkes.go.id/baca/blog/20240110/5344736/saatnya-mengatur-si-manis/)

Secara global, pada tahun 2024, diperkirakan terdapat sekitar 589 juta orang dewasa (usia 20–79 tahun) yang hidup dengan diabetes, dan jumlah ini diprediksi akan mencapai 853 juta pada tahun 2050. Fakta lainnya yang mencengangkan adalah bahwa lebih dari 4 dari 5 orang dewasa (sekitar 81%) yang hidup dengan diabetes tinggal di negara-negara berpenghasilan rendah dan menengah, termasuk Indonesia. [[2]](https://diabetesatlas.org/data-by-location/global/)

Di Indonesia sendiri, pada tahun 2021, tercatat sekitar 19,5 juta penderita diabetes. Angka ini diprediksi meningkat menjadi 28,6 juta pada tahun 2045, menempatkan Indonesia sebagai negara dengan jumlah penderita diabetes kelima terbanyak di dunia . Persoalan ini menjadi perhatian serius dari Kementerian Kesehatan karena diabetes melitus merupakan induk dari berbagai penyakit kronis lainnya yang dapat menurunkan kualitas hidup secara signifikan. [[1]](http://sehatnegeriku.kemkes.go.id/baca/blog/20240110/5344736/saatnya-mengatur-si-manis/)

Berdasarkan latar belakang tersebut, proyek ini bertujuan untuk membangun sebuah *model machine learning* yang dapat memprediksi status diabetes seseorang berdasarkan data klinis dan demografis. Dengan adanya model prediksi ini, diharapkan dapat mendukung deteksi dini diabetes dan membantu pengambilan keputusan di sektor kesehatan preventif, khususnya di negara berkembang seperti Indonesia.

# Business Understanding

## Problem Statements

Berdasarkan latar belakang yang telah dijelaskan di atas, maka diperoleh rumusan masalah yang akan diselesaikan pada proyek ini, yaitu:
1. Bagaimana cara melakukan tahap persiapan data sebelum digunakan untuk membuat model *machine learning* untuk prediksi diabetes?
    
2. Bagaimana cara membuat model *machine learning* untuk memprediksi apakah seseorang menderita diabetes atau tidak berdasarkan data klinis dan demografis?

## Goals

Berdasarkan rumusan masalah yang telah dipaparkan di atas, maka didapatkan tujuan dari proyek ini, yaitu:
1. Melakukan tahap persiapan data (*data preparation*) sehingga data klinis dan demografis dapat digunakan dengan baik pada model *machine learning*.
2. Membuat model *machine learning* untuk melakukan analisis prediksi status diabetes dengan tingkat *error* yang rendah dan akurasi yang optimal

## Solution Statements
Terdapat beberapa solusi yang dapat dilakukan untuk dapat mencapai tujuan dari proyek ini, yaitu:
1. Tahap persiapan data (*data preparation*) dilakukan dengan beberapa langkah berikut:
   - Membagi dataset menjadi dua bagian, yaitu *training data* dan *testing data* dengan rasio 90:10, untuk keperluan pelatihan dan pengujian model.
   - Melakukan *standarisasi* pada fitur numerik seperti `age`, `bmi`, `hbA1c_level`, dan `blood_glucose_level` guna menghindari dominasi fitur tertentu karena skala yang berbeda.
   - Melakukan *encoding* pada fitur kategorikal seperti `gender`, `smoking_history`, `race`, dan `location` agar bisa diproses oleh algoritma *machine learning*.

2. Tahap pembangunan model *machine learning* dilakukan dengan menggunakan tiga algoritma berbeda adalah K-Nearest Neighbor (KNN), Random Forest, dan Adaptive Boosting (AdaBoost). Seluruh model akan dievaluasi menggunakan metrik seperti *Mean Squared Error (MSE)* dan akurasi, lalu dibandingkan untuk menentukan algoritma terbaik dalam memprediksi status diabetes berdasarkan data klinis dan demografis.
	- **Algoritma K-Nearest Neighbor**
	Sesuai dengan namanya, yaitu "sejumlah k-tetangga terdekat", **K-Nearest Neighbor (KNN)** adalah algoritma _machine learning_ yang termasuk dalam kategori *supervised learning*. Algoritma ini bekerja dengan cara mengelompokkan data berdasarkan kemiripan antara data baru dan sejumlah data (`k`) terdekat di dalam dataset. [[3]](https://dqlab.id/fleksibilitas-kunci-keunggulan-algoritma-k-nearest-neighbors-knn) 
Cara kerja algoritma K-Nearest Neighbor adalah sebagai berikut: 
		-   Menentukan jumlah tetangga terdekat (`k`) yang akan digunakan dalam proses klasifikasi.
	    -   Menghitung jarak antara data baru terhadap seluruh titik data dalam dataset. Jarak yang umum digunakan meliputi **Euclidean Distance**, **Manhattan Distance**, dan **Minkowski Distance**.
		-   Mengurutkan hasil perhitungan jarak dari yang paling dekat ke yang paling jauh.  
		-   Mengambil sejumlah `k` titik dengan jarak terdekat.
	    -   Menentukan kelas dari data baru berdasarkan mayoritas kelas dari `k` tetangga terdekat tersebut (untuk klasifikasi).  Jika digunakan untuk regresi, nilai prediksi ditentukan berdasarkan **rata-rata nilai** dari tetangga-tetangga tersebut.
	   <br>
	   <img src="https://cdn-images-1.medium.com/v2/resize:fit:800/0*QmLAPLYUDcpJYwvo.png" alt="Ilustrasi Algoritma K-Nearest Neighbor" title="Ilustrasi Algoritma K-Nearest Neighbor">
	   
	   Perhitungan jarak ke tetangga terdekat dapat dilakukan dengan menggunakan metrik sebagai berikut:[[4]](https://www.ibm.com/id-id/think/topics/knn)
	     - *Euclidean distance*
       $$d(x,y)=\sqrt{\sum_{i=1}^n (x_i-y_i)^2}$$
	     - *Manhattan distance*
       $$d(x,y)=\sum_{i=1}^n |x_i-y_i|$$
	     - *Hamming distance*
       $$d(x,y)=\frac{1}{n}\sum_{n=1}^{n=n} |x_i-y_i|$$
	     - *Minkowski distance*
       $$d(x,y)=\left(\sum_{i=1}^n |x_i-y_i|^p\right)^\frac{1}{p}$$
		
		Adapun kelebihan dari algoritma K-Nearest Neighbor, yaitu: [[5]](https://www.trivusi.web.id/2022/06/algoritma-knn.html)
		- Mudah untuk dipahami dan diterapkan, sehingga sangat cocok untuk pemula dalam machine learning.
		- Mudah menyesuaikan diri dengan perubahan dalam dataset. 
		- KNN hanya memiliki dua hyperparameter utama: nilai K (jumlah tetangga terdekat yang akan diperhitungkan) dan metrik jarak yang digunakan. 

		Adapun kelemahan dari algoritma K-Nearest Neighbor, yaitu: [[5]](https://www.trivusi.web.id/2022/06/algoritma-knn.html) 
		- Tidak cocok untuk dataset berukuran besar. 
		- Tidak cocok untuk dimensi tinggi.
		- Tanpa penskalaan yang benar ([standarisasi dan normalisasi](https://www.trivusi.web.id/2022/09/normalisasi-data.html)), KNN dapat menghasilkan prediksi yang salah karena beberapa fitur memiliki skala yang dominan.
		- Sensitif terhadap noise, missing value, dan outlier dalam dataset. 
			
	- **Algoritma Random Forest**
	Random Forest adalah algoritma *machine learning* berbasis *ensemble learning* yang menggabungkan hasil dari banyak *decision tree* melalui proses *bagging*. Setiap pohon menghasilkan prediksi, dan hasil akhir ditentukan berdasarkan vote terbanyak. Semakin banyak jumlah pohon, semakin tinggi akurasi dan semakin kecil risiko *overfitting*.[[6]](https://www.trivusi.web.id/2022/08/algoritma-random-forest.html)
	
		<img src="https://www.researchgate.net/profile/Muhammad-Yaseen-Khan/publication/354354484/figure/fig4/AS:1080214163595269@1634554534720/Illustration-of-random-forest-trees.jpg" alt="Algoritma Random Forest" title="Algortima Random Forest"> 

		Cara kerja algoritma Random Forest :
		- Algoritma memilih sampel acak dari dataset yang disediakan.
		- Membuat decision tree untuk setiap sampel yang dipilih. Kemudian akan didapatkan hasil prediksi dari setiap decision tree yang telah dibuat.
		- Dilakukan proses voting untuk setiap hasil prediksi. Untuk masalah klasifikasi menggunakan modus (nilai yg paling sering muncul), sedangkan untuk masalah regresi akan menggunakan mean (nilai rata-rata).
		-  Algoritma akan memilih hasil prediksi yang paling banyak dipilih (vote terbanyak) sebagai prediksi akhir.[[6]](https://www.trivusi.web.id/2022/08/algoritma-random-forest.html)
		- Setelah dilakukan pelatihan, prediksi untuk sampel yang tidak terlihat ($x'$) dapat dibuat dengan menghitung rata-rata prediksi dari semua pohon setiap individu model pada $x'$. [[7]](https://en.wikipedia.org/wiki/Random_forest#Bagging 'Random Forest - Bagging')
     $$\hat{f}=\frac{1}{B}\sum_{b=1}^{B} f_b(x^{'})$$
		
	- **Algoritma Adaptive Boosting**
	AdaBoost (Adaptive Boosting) adalah algoritma *ensemble learning* yang melatih beberapa *weak learner* secara berulang, dengan memberi bobot lebih besar pada data yang salah diklasifikasikan. Setiap model diberi bobot berdasarkan performanya, lalu digabung menjadi satu model akhir. Proses ini membantu mengurangi bias dan variansi, serta banyak digunakan di bidang seperti computer vision, NLP, dan deteksi penipuan.[[8]](https://www.trivusi.web.id/2023/07/algoritma-adaboost.html)

	<img src="https://almablog-media.s3.ap-south-1.amazonaws.com/image_28_7cf514b000.png" alt="Algoritma AdaBoost" title="Algoritma AdaBoost"> 
	
	Algoritma AdaBoost mengacu kepada metode tertentu untuk melakukan pelatihan *classifier* yang di-*boosted*. Pengklasifikasian tersebut adalah pengklasifikasian dalam bentuk, 
     $$F_T(x)=\sum_{t=q}^{T}f_t(x)$$
     di mana setiap $F_T$ adalah *learner* yang lemah yang mengambil objek $x$ sebagai input dan mengembalikan nilai yang menunjukkan kelas objek. Demikian juga pada pengklasifikasi $T$ merupakan nilai positif jika sampel berada dalam kelas positif, dan negatif jika sebaliknya. [[9]](https://en.wikipedia.org/wiki/AdaBoost#Training 'AdaBoost - Training')

## Data Understanding

Data yang digunakan dalam proyek ini adalah  _dataset_  yang diambil dari Kaggle Dataset  [Diabetes Clinical Dataset(100k rows)](https://www.kaggle.com/datasets/ziya07/diabetes-clinical-dataset100k-rows/data)  dengan kategori  _dataset_, yaitu  _Health Conditions_  dan  _Diabetes_. Dalam  _dataset_  tersebut terdapat sebuah  _file_  atau berkas dengan nama  `diabetes_dataset_with_notes.csv`  yang berekstensi (_file format_)  `.csv`  atau  [comma-separated values](https://en.wikipedia.org/wiki/Comma-separated_values "Comma-separated values")  berukuran 26,94 MB.

Dari  _dataset_  tersebut, masih perlu dilakukan penyesuaian hingga  _dataset_  dapat benar-benar digunakan. Beberapa penyesuaian tersebut, yaitu
 - Menggabungkan kolom `race:AfricanAmerican`, `race:Asian`, `race:Caucasian`, `race:Hispanic`, `race:Other` menjadi satu kolom baru `race`.
 ```python
 race_columns = ['race:AfricanAmerican', 'race:Asian', 'race:Caucasian', 'race:Hispanic', 'race:Other']
diabet['race'] = diabet[race_columns].idxmax(axis=1)
diabet['race'] = diabet['race'].str.replace('race:', '')
diabet.drop(columns=race_columns, inplace=True)
```
 - Menghapus kolom yang tidak akan digunakan yaitu, kolom `clinical_notes`.
 ``` python
 diabet.drop('clinical_notes', inplace=True, axis=1)
 ```
 
 Kemudian dilakukan proses *Exploratory Data Analysis* (EDA) sebagai investigasi awal untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data dengan menggunakan teknik statistik dan representasi grafis atau visualisasi.

1. **Deskripsi Variabel**  
   Berikut adalah informasi mengenai variabel-variabel yang terdapat pada *dataset* *Diabet* adalah sebagai berikut,
   
<img src="https://raw.githubusercontent.com/addsarah/diabet-predictive-analytics/refs/heads/main/img/Deskripsi%20Variabel.png" alt="Deskripsi Variabel" title="Deskripsi Variabel">

Dari gambar di atas dapat dilihat bahwa terdapat 100.000 baris data dan 12 kolom atribut atau fitur. Di antaranya adalah tiga (3) atribut dengan tipe data `float64` dan lima (5) atribut dengan tipe data `int64`. Selain itu, terdapat empat (4) atribut dengan tipe data `object` yang merepresentasikan data kategorikal seperti `gender`, `location`, `smooking_history`, dan `race`. Berikut adalah keterangan untuk masing-masing variabel,
- `year` : Tahun pencatatan data dilakukan.
- `gender` : Jenis kelamin pasien (contoh: Male, Female, Other).
- `age` : Usia pasien dalam tahun.
- `location` : Lokasi atau wilayah tempat pasien berada.
- `hypertension` : Status hipertensi pasien (0 = tidak, 1 = ya).
- `heart_disease` : Status riwayat penyakit jantung pasien (0 = tidak, 1 = ya).
- `smoking_history` : Riwayat merokok pasien (contoh: never, current, former, no_info).
- `bmi` : Body Mass Index (Indeks Massa Tubuh) pasien.
- `hbA1c_level` : Kadar hemoglobin A1c (%), indikator kadar gula darah jangka panjang.
- `blood_glucose_level` : Kadar glukosa dalam darah saat ini (mg/dL).
- `diabetes` : Status diabetes pasien (0 = tidak, 1 = ya).
- `race` : Kategori ras pasien (contoh: Asian, Hispanic, AfricanAmerican, dll).

2. **Deskripsi Statistik**
<img src="https://raw.githubusercontent.com/addsarah/diabet-predictive-analytics/refs/heads/main/img/Deskripsi%20Statistik.png" alt="Deskripsi Statistik" title="Deskripsi Statistik">

3. **Menangani Missing Value**

4. 
