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
 - Tahap persiapan data (*data preparation*) dilakukan dengan beberapa langkah berikut:
 - Membagi dataset menjadi dua bagian, yaitu *training data* dan *testing data* dengan rasio 90:10, untuk keperluan pelatihan dan pengujian model.
 - Melakukan *standarisasi* pada fitur numerik seperti `age`, `bmi`, `hbA1c_level`, dan `blood_glucose_level` guna menghindari dominasi fitur tertentu karena skala yang berbeda.
 - Melakukan *encoding* pada fitur kategorikal seperti `gender`, `smoking_history`, `race`, dan `location` agar bisa diproses oleh algoritma *machine learning*.

 - Tahap pembangunan model *machine learning* dilakukan dengan menggunakan tiga algoritma berbeda adalah K-Nearest Neighbor (KNN), Random Forest, dan Adaptive Boosting (AdaBoost). Seluruh model akan dievaluasi menggunakan metrik seperti *Mean Squared Error (MSE)* dan akurasi, lalu dibandingkan untuk menentukan algoritma terbaik dalam memprediksi status diabetes berdasarkan data klinis dan demografis.
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

		Adapun kekurangan dari algoritma K-Nearest Neighbor, yaitu: [[5]](https://www.trivusi.web.id/2022/06/algoritma-knn.html) 
		- Tidak cocok untuk dataset berukuran besar. 
		- Tidak cocok untuk dimensi tinggi.
		- Tanpa penskalaan yang benar ([standarisasi dan normalisasi](https://www.trivusi.web.id/2022/09/normalisasi-data.html)), KNN dapat menghasilkan prediksi yang salah karena beberapa fitur memiliki skala yang dominan.
		- Sensitif terhadap noise, missing value, dan outlier dalam dataset. 
			
	- **Algoritma Random Forest**
	Random Forest adalah algoritma *machine learning* berbasis *ensemble learning* yang menggabungkan hasil dari banyak *decision tree* melalui proses *bagging*. Setiap pohon menghasilkan prediksi, dan hasil akhir ditentukan berdasarkan vote terbanyak. Semakin banyak jumlah pohon, semakin tinggi akurasi dan semakin kecil risiko *overfitting*.[[6]](https://www.trivusi.web.id/2022/08/algoritma-random-forest.html)
	
		<img src="https://miro.medium.com/v2/resize:fit:1400/1*wDKMZMDywDP_iQ0ZiXr34g.jpeg" alt="Algoritma Random Forest" title="Algortima Random Forest"> 

		Cara kerja algoritma Random Forest :
		- Algoritma memilih sampel acak dari dataset yang disediakan.
		- Membuat decision tree untuk setiap sampel yang dipilih. Kemudian akan didapatkan hasil prediksi dari setiap decision tree yang telah dibuat.
		- Dilakukan proses voting untuk setiap hasil prediksi. Untuk masalah klasifikasi menggunakan modus (nilai yg paling sering muncul), sedangkan untuk masalah regresi akan menggunakan mean (nilai rata-rata).
		-  Algoritma akan memilih hasil prediksi yang paling banyak dipilih (vote terbanyak) sebagai prediksi akhir.[[6]](https://www.trivusi.web.id/2022/08/algoritma-random-forest.html)
		- Setelah dilakukan pelatihan, prediksi untuk sampel yang tidak terlihat ($x'$) dapat dibuat dengan menghitung rata-rata prediksi dari semua pohon setiap individu model pada $x'$. 
     $$\hat{f}=\frac{1}{B}\sum_{b=1}^{B} f_b(x^{'})$$ [[7]](https://www.ibm.com/think/topics/random-forest)

    	Kelebihan dari Algoritma Random Forest, yaitu:[[8]](https://dqlab.id/serba-serbi-machine-learning-model-random-forest)
		- Dengan menggunakan teknik bootstrap aggregating (bagging) dan pemilihan fitur acak, Random Forest dapat mengurangi varians dalam model dan meningkatkan generalisasi pada data baru.
		-  Random Forest menggabungkan beberapa pohon keputusan yang independen, ia memiliki kemampuan yang lebih baik untuk menangani noise dan variasi dalam data, sehingga menghasilkan prediksi yang lebih konsisten dan andal.
		- Dengan menganalisis sejauh mana setiap fitur berkontribusi dalam memisahkan kelas target, Random Forest dapat memberikan insight berguna tentang fitur mana yang paling informatif.
		- Dalam Random Forest, pengambilan sampel acak dan pembobotan kelas dapat digunakan untuk mengurangi efek ketidakseimbangan data dan menghasilkan prediksi yang lebih akurat untuk kelas yang kurang representatif.
    
   		Kekurangan Algoritma Random Forest, yaitu:[[8]](https://dqlab.id/serba-serbi-machine-learning-model-random-forest)
		- Random Forest memiliki kemampuan untuk menghasilkan prediksi yang akurat. Namun, interpretabilitas model bisa menjadi tantangan. Karena Random Forest terdiri dari banyak pohon keputusan yang digabungkan, sehingga sulit untuk secara langsung menginterpretasikan pengaruh setiap fitur dalam membuat keputusan. 
		- Random Forest memiliki beberapa parameter yang perlu dikonfigurasi, seperti jumlah pohon dalam ensambel, jumlah fitur yang diambil secara acak, dan kriteria pemisahan pada setiap simpul pohon. Memilih parameter yang optimal dapat memerlukan eksperimen dan tuning yang cermat, terutama jika tidak ada pengetahuan sebelumnya tentang dataset yang digunakan.
		- Random Forest mungkin menghadapi tantangan dalam menangani data dengan dimensi yang sangat tinggi, yaitu ketika jumlah fitur sangat besar dibandingkan dengan jumlah sampel data yang tersedia. Dalam kasus seperti itu, pengambilan sampel acak dan pemilihan fitur acak mungkin tidak efektif dalam mengekstraksi pola yang relevan dan menghasilkan prediksi yang akurat.

		
	- **Algoritma Adaptive Boosting**
	AdaBoost (Adaptive Boosting) adalah algoritma *ensemble learning* yang melatih beberapa *weak learner* secara berulang, dengan memberi bobot lebih besar pada data yang salah diklasifikasikan. Setiap model diberi bobot berdasarkan performanya, lalu digabung menjadi satu model akhir. Proses ini membantu mengurangi bias dan variansi, serta banyak digunakan di bidang seperti computer vision, NLP, dan deteksi penipuan.[[9]](https://www.trivusi.web.id/2023/07/algoritma-adaboost.html)

	<img src="https://almablog-media.s3.ap-south-1.amazonaws.com/image_28_7cf514b000.png" alt="Algoritma AdaBoost" title="Algoritma AdaBoost"> 
	

	Algoritma AdaBoost mengacu kepada metode tertentu untuk melakukan pelatihan *classifier* yang di-*boosted*. Pengklasifikasian tersebut adalah pengklasifikasian dalam bentuk, 
     $$F_T(x)=\sum_{t=q}^{T}f_t(x)$$
     di mana setiap $F_T$ adalah *learner* yang lemah yang mengambil objek $x$ sebagai input dan mengembalikan nilai yang menunjukkan kelas objek. Demikian juga pada pengklasifikasi $T$ merupakan nilai positif jika sampel berada dalam kelas positif, dan negatif jika sebaliknya. [[10]](https://aws.amazon.com/id/what-is/boosting/)
     

   	Adapun Kelebihan dari Algoritma Adaptive Boosting, yaitu:[[9]](https://www.trivusi.web.id/2023/07/algoritma-adaboost.html)

   	- Adaboost dapat secara signifikan meningkatkan akurasi prediksi dalam pemodelan machine learning. Dengan menggabungkan weak learners menjadi strong learner, Adaboost dapat mengatasi kesalahan klasifikasi dan meningkatkan kemampuan prediksi model.
	 - Adaboost efektif dalam menangani data yang kompleks dan memiliki interaksi fitur yang rumit. Dalam kasus di mana hubungan antara fitur-fitur input dan variabel output tidak sederhana, Adaboost dapat menangkap pola yang lebih kompleks daripada weak learners individu.
	 - Dengan memberikan bobot pada contoh-contoh yang salah diklasifikasikan, Adaboost dapat mengurangi risiko overfitting. Hal ini membantu model untuk tidak terlalu fokus pada contoh-contoh pelatihan yang sulit dan memperbaiki generalisasi pada data uji.
	 

	Adapun kekurangan dari Algoritma Adaptive Boosting, yaitu:[[9]](https://www.trivusi.web.id/2023/07/algoritma-adaboost.html)
	-  Adaboost cenderung sensitif terhadap data yang mengandung noise atau outlier. Kehadiran contoh-contoh yang tidak representatif atau gangguan dapat mempengaruhi pembelajaran dan menghasilkan model yang kurang akurat.
	 - Jika dataset pelatihan sangat kecil, terdapat risiko overfitting pada Adaboost. Model yang terlalu kompleks dapat dengan mudah "menghafal" contoh-contoh pelatihan dan gagal dalam generalisasi pada data baru.
	 - Adaboost melibatkan iterasi berulang untuk melatih weak learners dan memperbarui bobot contoh-contoh pelatihan. Oleh karena itu, waktu pelatihan algoritma ini cenderung lebih lama dibandingkan dengan beberapa algoritma machine learning lainnya.

## Data Understanding
<img src="https://raw.githubusercontent.com/addsarah/diabet-predictive-analytics/refs/heads/main/img/Diabetes%20Clinical%20Dataset%20Kaggle%20Dataset.png" alt="Diabetes Clinical Dataset Kaggle Dataset" title="Diabetes Clinical Dataset Kaggle Dataset">

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

|                       | year      | age       | hypertension | heart_disease | bmi       | hbA1c_level | blood_glucose_level | diabetes  |
|-----------------------|-----------|-----------|--------------|----------------|-----------|-------------|----------------------|-----------|
| **count**             | 100000.00 | 100000.00 | 100000.00    | 100000.00      | 100000.00 | 100000.00   | 100000.00            | 100000.00 |
| **mean**              | 2018.36   | 41.89     | 0.0748       | 0.0394         | 27.32     | 5.53        | 138.06               | 0.0850    |
| **std**               | 1.35      | 22.52     | 0.2632       | 0.1946         | 6.64      | 1.07        | 40.71                | 0.2788    |
| **min**               | 2015.00   | 0.08      | 0.00         | 0.00           | 10.01     | 3.5         | 80.00                | 0.00      |
| **25%**               | 2019.00   | 24.00     | 0.00         | 0.00           | 23.63     | 4.8         | 100.00               | 0.00      |
| **50% (median)**      | 2019.00   | 43.00     | 0.00         | 0.00           | 27.32     | 5.8         | 140.00               | 0.00      |
| **75%**               | 2019.00   | 60.00     | 0.00         | 0.00           | 29.58     | 6.2         | 159.00               | 0.00      |
| **max**               | 2022.00   | 80.00     | 1.00         | 1.00           | 95.69     | 9.0         | 300.00               | 1.00      |


3. **Menangani Missing Value**

|                     | 0 |
|---------------------|----------------|
| year                | 0              |
| gender              | 0              |
| age                 | 0              |
| location            | 0              |
| hypertension        | 0              |
| heart_disease       | 0              |
| smoking_history     | 0              |
| bmi                 | 0              |
| hbA1c_level         | 0              |
| blood_glucose_level | 0              |
| diabetes            | 0              |
| race                | 0              |
| **dtype**           | **int64**      |

   Berdasarkan tabel tersebut, tidak terdapat *missing value*.
<br>

4. **Menangani *Outliers***

	*Outliers* merupakan sampel data yang nilainya berada sangat jauh dari cakupan umum data utama yang dapat merusak hasil analisis data. 

	Memisahkan data pasien dengan label diabetes dan non diabetes
	```python
	df_diabetes = diabet[diabet['diabetes'] == 1]
	df_ndiabetes = diabet[diabet['diabetes'] == 0]
	```
	
| diabetes | count |
|----------|-------|
| 0        | 91500 |
| 1        | 8500  |


	
 Berikut adalah visualisasi *boxplot* untuk melakukan pengecekan keberadaan *outliers* pada `diabetes` 
<img src="https://raw.githubusercontent.com/addsarah/diabet-predictive-analytics/refs/heads/main/img/Menangani%20Outliers%20Diabetes%20-%20Sebelum.png" alt="Menangani Outliers Diabetes - Sebelum" title="Menangani Outliers Diabetes - Sebelum"> 
	
 Berdasarkan gambar tersebut, terdapat *outliers* pada fitur `year`, `age`, `hypertension`, `heart_disease`, dan `bmi`. 


	
 `non-diabetes`
<img src="https://raw.githubusercontent.com/addsarah/diabet-predictive-analytics/refs/heads/main/img/Menangani%20Outliers%20Non%20Diabetes%20-%20Sebelum.png" alt="Menangani Outliers Non Diabetes - Sesudah" title="Menangani Outliers Non Diabetes - Sesudah">

Berdasarkan gambar tersebut, terdapat *outliers* pada fitur `year`, `hypertension`, `heart_disease`, dan `bmi`. 

Sehingga dilakukan proses pembersihan *outliers* dengan metode IQR (*Inter Quartile Range*). 

$$IQR=Q_3-Q_1$$
   
Kemudian membuat batas bawah dan batas atas untuk mencakup *outliers* dengan menggunakan,

   $BatasBawah=Q_1-1.5*IQR$
   
   $BatasAtas=Q_3-1.5*IQR$
   
   Setelah dilakukan pembersihan *outliers*, dilakukan kembali visualisasi *outliers* untuk melakukan pengecekan kembali sebagai berikut,

`diabetes`
<img src="https://raw.githubusercontent.com/addsarah/diabet-predictive-analytics/refs/heads/main/img/Menangani%20Outliers%20Diabetes%20-%20Sesudah.png" alt="Menangani Outliers Diabetes - Sesudah" title="Menangani Outliers Diabetes - Sesudah">

`non-diabetes`
<img src="https://raw.githubusercontent.com/addsarah/diabet-predictive-analytics/refs/heads/main/img/Menangani%20Outliers%20Non%20Diabetes%20-%20Sesudah.png" alt="Menangani Outliers Non Diabetes - Sesudah" title="Menangani Outliers Non Diabetes - Sesudah">

   Setelah dilakukan pembersihan *outliers* menggunakan metode IQR (*Inter Quartile Range*), dapat dilihat bahwa *outliers* telah berkurang pada boxplot di atas. Meskipun *outliers* masih ada pada fitur `age` dan`bmi`, tetapi masih dalam batas aman.
	
   Menggabungkan data diabetes dan non diabetes
```python
	df_diabet = pd.concat([df_diabetes, df_ndiabetes])
```
    
5. **Univariate Analysis**  
   Melakukan proses analisis data *univariate* pada fitur-fitur numerik. Proses analisis ini menggunakan bantuan visualisasi histogram untuk masing-masing fitur numerik dan kategorikal.

	<img src="https://raw.githubusercontent.com/addsarah/diabet-predictive-analytics/refs/heads/main/img/Univariate%20Analysis%20Gender.png" alt="Univariate Analysis Gender" title="Univariate Analysis Gender">

	Mayoritas pasien berjenis kelamin perempuan (58.7%), diikuti oleh laki-laki (41.3%), dan hanya 0.0% atau 16 orang yang dikategorikan sebagai Other

	<br>
	<img src="https://raw.githubusercontent.com/addsarah/diabet-predictive-analytics/refs/heads/main/img/Univariate%20Analysis%20Location.png" alt="Univariate Analysis Location" title="Univariate Analysis Location">
	
 	Distribusi pasien tersebar cukup merata di seluruh negara bagian AS. Lokasi dengan jumlah pasien tertinggi antara lain Nebraska, New Jersey, dan North Dakota, masing-masing menyumbang sekitar 2.1% dari total data. Sementara itu, lokasi dengan jumlah pasien paling sedikit adalah Wisconsin (hanya 2 sampel atau 0.0%) dan Virgin Islands (0.9%). Hal ini menunjukkan bahwa sebagian besar data berasal dari wilayah yang beragam, namun terdapat perbedaan jumlah sampel yang signifikan antar lokasi.
	
<br>
	<img src="https://raw.githubusercontent.com/addsarah/diabet-predictive-analytics/refs/heads/main/img/Univariate%20Analysis%20Smoking%20History.png" alt="Univariate Analysis Smoking History" title="Univariate Analysis Smoking History">
	
 Mayoritas data tidak memiliki informasi riwayat merokok (No Info) sebanyak 38.2%, diikuti oleh pasien yang tidak pernah merokok (never) sebesar 34.5%. Sementara itu, kategori lainnya seperti current (masih merokok), former (mantan perokok), not current, dan ever (pernah merokok) memiliki proporsi yang jauh lebih kecil, masing-masing kurang dari 10%. Hal ini menunjukkan bahwa lebih dari setengah data berasal dari pasien yang tidak pernah merokok atau tidak memiliki informasi lengkap terkait kebiasaan merokok.

<br>
	<img src="https://raw.githubusercontent.com/addsarah/diabet-predictive-analytics/refs/heads/main/img/Univariate%20Analysis%20Race.png" alt="Univariate Analysis Race" title="Univariate Analysis Race">
	
 Setiap kategori ras seperti AfricanAmerican, Caucasian, Hispanic, Asian, dan Other memiliki jumlah sampel yang hampir sama, dengan masing-masing berkisar 19.8% hingga 20.1%. Hal ini menunjukkan bahwa data telah dikonstruksi secara proporsional antar kelompok ras, sehingga tidak terjadi ketimpangan distribusi yang dapat memengaruhi performa model secara bias terhadap kelompok tertentu.

<br>
	<img src="https://raw.githubusercontent.com/addsarah/diabet-predictive-analytics/refs/heads/main/img/Univariate%20Analysis.png" alt="Univariate Analysis" title="Univariate Analysis">
	Dari data histogram di atas diperoleh informasi, yaitu:

 - `year` menunjukkan seluruh data diambil mayoritas pada tahun 2019.

 - `age` memiliki distribusi yang relatif merata dari usia muda hingga tua, tetapi terdapat lonjakan signifikan pada usia sekitar 80 tahun, menandakan banyaknya pasien berusia lanjut.

 - `hypertension` didominasi oleh nilai 0, yang berarti sebagian besar pasien tidak memiliki riwayat hipertensi.

 - `heart_disease` juga sangat didominasi oleh nilai 0, menunjukkan bahwa sebagian besar pasien tidak memiliki riwayat penyakit jantung.

 - `bmi` memiliki distribusi yang mendekati normal, namun terdapat lonjakan tinggi di satu nilai (sekitar 28–30).

 - `hbA1c_level` memiliki persebaran data yang beragam, tetapi paling sering berada pada level 6, yang merupakan batas prediabetes.

 - `blood_glucose_level` menunjukkan beberapa lonjakan pada nilai tertentu (sekitar 100–150), menunjukkan data tidak tersebar merata dan cenderung dikelompokkan.

 - `diabetes` sangat tidak seimbang, dimana mayoritas data menunjukkan pasien tidak menderita diabetes (nilai 0), dan hanya sebagian kecil yang menderita diabetes (nilai 1).


 6. **Multivariate Analysis**  
   Melakukan visualisasi distribusi data pada fitur-fitur numerik dari *dataframe* `diabet`. Visualisasi dilakukan dengan bantuan *library* `seaborn` `pairplot` menggunakan parameter `diag_kind`, yaitu `kde`, untuk melihat perkiraan distribusi probabilitas antar fitur numerik.
   <img src="https://raw.githubusercontent.com/addsarah/diabet-predictive-analytics/refs/heads/main/img/Multivariate%20Analysis.png" alt="Multivariate Analysis" title="Multivariate Analysis">
   
7. **Correlation Matrix with Heatmap**

	Melakukan pengecekan korelasi antar fitur numerik dengan menggunakan visualisasi diagram *heatmap* *correlation matrix*.
 <img src="https://raw.githubusercontent.com/addsarah/diabet-predictive-analytics/refs/heads/main/img/Correlation%20Matrix%20with%20Heatmap.png" alt="Correlation Matrix with Heatmap" title="Correlation Matrix with Heatmap">

   Dapat dilihat pada diagram *heatmap* di atas memiliki *range* atau rentang angka dari 1.0 hingga 0.08 dengan keterangan sebagai berikut,

   - Jika semakin mendekati 1, maka korelasi antar fitur numerik semakin kuat bernilai positif.
     
   - Jika semakin mendekati 0, maka korelasi antar fitur numerik semakin rendah.
     
   - Jika semakin mendekati -1, maka korelasi antar fitur numerik semakin kuat bernilai negatif.
   
   Jika korelasi bernilai positif, berarti nilai kedua fitur numerik cenderung meningkat bersama-sama.  
   
   Jika korelasi bernilai negatif, berarti nilai salah satu fitur numerik cenderung meningkat ketika nilai fitur numerik yang lain menurun.
   
## Data Preparation

Pada tahap persiapan data atau *data preparation* dilakukan berdasarkan penjelasan yang sudah dipaparkan pada bagian [Solution Statements](#solution-statements "Solution Statements"). Tahap ini penting dilakukan untuk mempersiapkan data sehingga dapat digunakan untuk melatih model *machine learning* dengan baik. Berikut adalah dua tahapan data preparation yang dilakukan, yaitu,

1. **Split Data**  
   Mengubah fitur kategorikal menjadi numerik menggunakan Label Encoding agar semua data dalam dataset bertipe numerik.
   ```python
   categorical_features = ['gender', 'location', 'smoking_history', 'race']
   le = LabelEncoder()
   df_diabet[feature] = le.fit_transform(df_diabet[feature])
   df_diabet.head()
   ```
   Pembagian data dilakukan untuk memisahkan data keseluruhan menjadi dua (2) bagian, yaitu data latih (*training data*) dan data uji (*testing data*) dengan perbandingan rasio sebesar 90 : 10 menggunakan `train_test_split`.
	```python
	xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.1, random_state=123)
	```   

	Kemudian diperoleh hasil pembagian data masing-masing, yaitu sebagai berikut,
	```python
	Total seluruh sampel : 66511 
	Total data train : 59859 
	Total data test : 6652
	```
2. **Standarisasi pada Fitur Numerik**  
   Standarisasi fitur numerik menggunakan `StandardScaler` untuk mencegah terjadinya penyimpangan nilai data yang cukup besar. Proses standarisasi tersebut dilakukan dengan mengurangkan nilai rata-rata, lalu membaginya dengan standar deviasi atau simpangan baku untuk menggeser distribusi. Proses standarisasi akan menghasilkan distribusi dengan nilai rata-rata menjadi 0, dan nilai standar deviasi menjadi 1.
	```python
	scaler = StandardScaler()
	scaler.fit(xTrain[numericalFeatures])
	xTrain[numericalFeatures] = scaler.transform(xTrain.loc[:, numericalFeatures])
	```


|       | age      | bmi      | hbA1c_level | blood_glucose_level |
|-------|----------|----------|-------------|----------------------|
| 57684 | -1.159412 | -1.056342 | 0.576090    | 0.462202             |
| 51469 | -0.028597 | 0.082958  | -0.475589   | -0.941541            |
| 59092 | -0.933249 | -0.899889 | 0.671697    | 0.206976             |
| 72187 | 0.559428  | 1.129990  | 0.193661    | 0.462202             |
| 34182 | 0.876056  | 0.227376  | 0.480483    | 0.538770             |




```python
	xTrain[numericalFeatures].describe().round(4)
```
	
	
|                       | age      | bmi      | hbA1c_level | blood_glucose_level |
|-----------------------|----------|----------|-------------|----------------------|
| **count**             | 59859.000 | 59859.000 | 59859.000    | 59859.000             |
| **mean**              | -0.0000   | 0.0000    | 0.0000       | -0.0000               |
| **std**               | 1.0000    | 1.0000    | 1.0000       | 1.0000                |
| **min**               | -1.7891   | -2.2839   | -1.9097      | -1.4520               |
| **25%**               | -0.7976   | -0.6191   | -0.6668      | -0.9415               |
| **50% (median)**      | -0.0286   | 0.2274    | 0.2893       | 0.0794                |
| **75%**               | 0.7856    | 0.3517    | 0.6717       | 0.5643                |
| **max**               | 1.8259    | 4.5158    | 3.3487       | 4.1630                |


## Modelling

Setelah dilakukannya tahap *data preparation*, selanjutnya adalah melakukan tahap persiapan model terlebih dahulu sebelum mengembangkan model menggunakan algoritma yang telah ditentukan.

Tahap persiapan *dataframe* untuk analisis model menggunakan parameter `index`, yaitu train_mse dan test_mse, serta parameter `columns` yang merupakan algoritma yang akan digunakan untuk melakukan prediksi, yaitu algoritma K-Nearest Neighbor (KNN), Random Forest, dan Adaptive Boosting (AdaBoost).
```python
models = pd.DataFrame(
    index   = ['train_mse', 'test_mse'],
    columns = ['KNN', 'RandomForest', 'Boosting']
)
```

Kemudian terapkan ketiga algoritma ke dalam model tersebut.

1. **K-Nearest Neighbor (KNN) Algorithm**  
   Pada algoritma K-Nearest Neighbor digunakan parameter `n_neighbors` dengan nilai k = 10 tetangga dan `metric` bawaan, yaitu Euclidean.
   
   ```python
   knn = KNeighborsRegressor(n_neighbors=10)
   ```
   
   Kemudian akan dilakukan analisis prediksi *error* menggunakan *Mean Squared Error* (MSE) pada data latih (*training data*) dan data uji (*testing data*)
   
2. **Random Forest Algorithm**  
   Pada algoritma K-Nearest Neighbor digunakan parameter `n_estimator` dengan jumlah 50 *trees* (pohon), `max_depth` dengan nilai kedalaman atau panjang pohon 16, `random_state` dengan nilai 55, dan `n_jobs` yang bernilai -1 (pekerjaan dilakukan secara paralel).
   
   ```python
   rf = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
   ```
   
   Kemudian akan dilakukan analisis prediksi *error* menggunakan *Mean Squared Error* (MSE) pada data latih (*training data*) dan data uji (*testing data*)
   
3. **Adaptive Boosting (AdaBoost) Algorithm**  
   Pada algoritma K-Nearest Neighbor digunakan parameter `learning_rate` dengan nilai bobot setiap *regressor* adalah 0.05, dan `random_state` dengan nilai 55.
   
   ```python
   boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
   ```
   
   Kemudian akan dilakukan analisis prediksi *error* menggunakan *Mean Squared Error* (MSE) pada data latih (*training data*) dan data uji (*testing data*)

Ketiga model yang telah dibangun di atas, akan dilakukan pengujian kinerja untuk masing-masing model yang menggunakan algoritma K-Nearest Neighbor, algoritma Random Forest, dan algoritma Adaptive Boosting. Dari ketiga model tersebut akan diperoleh satu (1) model dengan hasil prediksi yang paling baik dan tingkat *error* yang paling rendah.

## Evaluation

Pada tahap evaluasi model, akan dilakukan pengujian untuk melihat algoritma mana yang memberikan hasil prediksi paling baik dan dengan tingkat *error* yang paling rendah. Sebelumnya, akan dilakukan proses standarisasi atau *scaling* pada fitur numerik data uji (*testing data*) agar nilai rata-rata (*mean*) bernilai 0, dan varians bernilai 1.

```python
xTest.loc[:, numericalFeatures] = scaler.transform(xTest[numericalFeatures])
```

Kemudian evaluasi dari ketiga model, yaitu algoritma K-Nearest Neighbor, Random Forest, dan Adaptive Boosting (AdaBoost) untuk masing-masing data latih (*training data*) dan data uji (*testing data*) dengan melihat tingkat *error*-nya menggunakan *Mean Squared Error* (MSE),

$$MSE=\frac{1}{N}\sum_{i=1}^{N} (y_i-y\\_pred_i)^2$$

di mana, nilai $N$ adalah jumlah *dataset*, nilai $y_i$ merupakan nilai sebenarnya, dan $y\\_pred$ yaitu nilai prediksinya.

Penggunaan metode metrik *Mean Squared Error* (MSE) memiliki kelebihan, yaitu cukup sederhana dan mudah dipahami dalam melakukan perhitungan. Meskipun begitu, terdapat kelemahan pada metrik ini, yaitu hasil akurasi prediksi yang kecil karena tidak dapat membandingan hasil peramalan tersebut dengan kenyataannya. 

```python
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN', 'RF', 'Boosting'])
modelDict = {'KNN': knn, 'RF': rf, 'Boosting': boosting}
for name, model in modelDict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=yTrain, y_pred=model.predict(xTrain))/1e3
    mse.loc[name, 'test']  = mean_squared_error(y_true=yTest,  y_pred=model.predict(xTest))/1e3
```


|           | train     | test      |
|-----------|-----------|-----------|
| **KNN**   | 0.000035  | 0.000034  |
| **RF**    | 0.000016  | 0.000016  |
| **Boosting** | 0.000053  | 0.000048  |


Dari data tabel tersebut dapat divisualisasikan pada grafik batang berikut.
<img src="https://raw.githubusercontent.com/addsarah/diabet-predictive-analytics/refs/heads/main/img/Evaluation%20Graph.png" alt="Evaluation Graph" title="Evaluation Graph">

Dari visualisasi diagram di atas dapat disimpulkan bahwa,

1. Model dengan algoritma Random Forest memberikan nilai error yang paling kecil, yaitu sebesar 0.000016 pada training error, dan 0.000016 pada testing error. Ini menunjukkan bahwa model mampu melakukan prediksi dengan baik dan memiliki generalisasi yang kuat.

2. Model dengan algoritma K-Nearest Neighbor memiliki tingkat error yang sedang di antara dua algoritma lainnya, dengan nilai training error sebesar 0.000035, dan testing error sebesar 0.000034. Meskipun selisih error kecil, performa Boosting masih berada di bawah Adaptive Boosting dalam hal akurasi prediksi keseluruhan.

3. Model dengan algoritma Adaptive Boosting mengalami error yang lebih besar dibandingkan Random Forest dan K-Nearest Neighbor, dengan nilai training error sebesar 0.000053, dan testing error sebesar 0.000048.

Selanjutnya adalah pengujian prediksi dengan menggunakan beberapa nilai diabetes `df_diabet` dari data uji (*testing*)


|       | y_true | prediksi_KNN | prediksi_RF | prediksi_Boosting |
|-------|--------|---------------|-------------|--------------------|
| 86159 |   0    |       0       |      0      |         0          |


Dapat dilihat bahwa prediksi pada model dengan algoritma **Random Forest** memberikan hasil yang **paling mendekati nilai `y_true`** jika dibandingkan dengan model algoritma lainnya.
  
Nilai `y_true` adalah **0**, yang berarti pasien tersebut **tidak menderita diabetes**, dan ketiga model (`KNN`, `Random Forest`, dan `Boosting`) sama-sama menghasilkan prediksi sebesar **0.0** yang menunjukkan bahwa model memperkirakan pasien tidak mengidap diabetes.

Meskipun dalam kasus ini seluruh model memberikan hasil prediksi yang identik dengan nilai sebenarnya, model **Random Forest** tetap dianggap sebagai model dengan **tingkat *error* paling rendah secara keseluruhan**, berdasarkan evaluasi sebelumnya pada metrik MSE (*Mean Squared Error*).

Kesimpulannya adalah model yang digunakan untuk melakukan prediksi **diabetes (binary classification)** memberikan hasil terbaik ketika menggunakan algoritma **Random Forest**, karena mampu menghasilkan prediksi yang akurat dan konsisten dengan nilai `y_true`.

## Referensi
[1] Saatnya Mengatur Si Manis – Sehat Negeriku. (2024, January 10). Sehat Negeriku. Retrieved May 09, 2025, from https://sehatnegeriku.kemkes.go.id/baca/blog/20240110/5344736/saatnya-mengatur-si-manis/

[2] Global Diabetes Data & Insights. (n.d.). IDF Diabetes Atlas. Retrieved May 09, 2025, from https://diabetesatlas.org/data-by-location/global/

[3] Zuliasyari, L. (2025, February 10). Fleksibilitas: Kunci Keunggulan Algoritma K-Nearest Neighbors (KNN). DQLab. Retrieved May 09, 2025, from https://dqlab.id/fleksibilitas-kunci-keunggulan-algoritma-k-nearest-neighbors-knn

[4] Apa algoritma k-Nearest Neighbors? (n.d.). IBM. Retrieved May 09, 2025, from https://www.ibm.com/id-id/think/topics/knn

[5] Yuk Kenali Apa itu Algoritma K-Nearest Neighbors (KNN). (2023, September 24). Trivusi. Retrieved May 09, 2025, from https://www.trivusi.web.id/2022/06/algoritma-knn.html

[6] Algoritma Random Forest: Pengertian dan Kegunaannya. (2022, September 17). Trivusi. Retrieved May 09, 2025, from https://www.trivusi.web.id/2022/08/algoritma-random-forest.html

[7] What Is Random Forest? (n.d.). IBM. Retrieved May 09, 2025, from https://www.ibm.com/think/topics/random-forest

[8] Machine Learning Model & Tutorialnya Membangunnya. (2023, July 18). DQLab. Retrieved May 09, 2025, from https://dqlab.id/serba-serbi-machine-learning-model-random-forest

[9] Algoritma AdaBoost: Pengertian, Cara Kerja, dan Kegunaannya. (2023, July 17). Trivusi. Retrieved May 09, 2025, from https://www.trivusi.web.id/2023/07/algoritma-adaboost.html

[10] Apa itu Boosting? - Penjelasan tentang Boosting dalam Machine Learning? (n.d.). AWS. Retrieved May 09, 2025, from https://aws.amazon.com/id/what-is/boosting/





