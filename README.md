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
	Sesuai dengan namanya, yaitu "sejumlah k-tetangga terdekat", **K-Nearest Neighbor (KNN)** adalah algoritma _machine learning_ yang termasuk dalam kategori *supervised learning*. Algoritma ini bekerja dengan cara mengelompokkan data berdasarkan kemiripan antara data baru dan sejumlah data (`k`) terdekat di dalam dataset. [[3]](https://dqlab.id/fleksibilitas-kunci-keunggulan-algoritma-k-nearest-neighbors-knn). Cara kerja algoritma K-Nearest Neighbor adalah sebagai berikut: 
		-   Menentukan jumlah tetangga terdekat (`k`) yang akan digunakan dalam proses klasifikasi.
	    -   Menghitung jarak antara data baru terhadap seluruh titik data dalam dataset. Jarak yang umum digunakan meliputi **Euclidean Distance**, **Manhattan Distance**, dan **Minkowski Distance**.
		-   Mengurutkan hasil perhitungan jarak dari yang paling dekat ke yang paling jauh.  
		-   Mengambil sejumlah `k` titik dengan jarak terdekat.
	    -   Menentukan kelas dari data baru berdasarkan mayoritas kelas dari `k` tetangga terdekat tersebut (untuk klasifikasi).  Jika digunakan untuk regresi, nilai prediksi ditentukan berdasarkan **rata-rata nilai** dari tetangga-tetangga tersebut.
	   
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
		- Kemudahan implementasi, KNN dikenal karena kesederhanaannya. Algoritma ini mudah untuk dipahami dan diterapkan, sehingga sangat cocok untuk pemula dalam dunia data science dan machine learning.
		- Kemampuan beradaptasi, KNN dapat dengan mudah menyesuaikan diri dengan perubahan dalam dataset. Ketika sampel pelatihan baru ditambahkan, KNN akan langsung mempertimbangkan data baru ini karena semua data pelatihan disimpan dalam memori.
		- Hyperparameter yang sedikit, KNN hanya memiliki dua hyperparameter utama: nilai K (jumlah tetangga terdekat yang akan diperhitungkan) dan metrik jarak yang digunakan. Hal ini membuat proses penentuan parameter model menjadi lebih sederhana dan kurang kompleks dibandingkan dengan beberapa algoritma lainnya.

			Adapun kelemahan dari algoritma K-Nearest Neighbor, yaitu: [[5]](https://www.trivusi.web.id/2022/06/algoritma-knn.html) 
			- Tidak cocok untuk dataset berukuran besar, salah satu kelemahan utama KNN adalah ketidakcocokannya untuk dataset berukuran besar. Algoritma ini memerlukan perhitungan jarak antara titik baru dengan semua titik dalam dataset, sehingga biaya komputasi menjadi sangat besar dan dapat mengurangi kinerja algoritma secara signifikan pada dataset yang besar.
			-  Tidak cocok untuk dimensi tinggi, KNN tidak efektif pada data dengan dimensi tinggi. Ketika jumlah dimensi meningkat, algoritma akan menghadapi masalah perhitungan jarak yang semakin rumit dan memerlukan lebih banyak data untuk melakukan perhitungan yang akurat.
			- Penskalaan fitur diperlukan, sebelum menerapkan KNN pada dataset, penting untuk melakukan penskalaan fitur. Tanpa penskalaan yang benar ([standarisasi dan normalisasi](https://www.trivusi.web.id/2022/09/normalisasi-data.html)), KNN dapat menghasilkan prediksi yang salah karena beberapa fitur memiliki skala yang dominan.
			- Sensitif terhadap noise, missing value, dan outlier, KNN cenderung sensitif terhadap noise dalam dataset. Ini berarti kita perlu melakukan pemrosesan data yang cermat, termasuk mengatasi nilai yang hilang dan mengidentifikasi serta mengatasi outlier, sebelum menggunakan KNN.
			
	- **Algoritma Random Forest**
	- **Algoritma Adaptive Boosting**
