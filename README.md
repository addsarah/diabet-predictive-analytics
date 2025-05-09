# Laporan Proyek Pertama Machine Learning Expert Dicoding: Predictive Analytics - Diabet (split diabet n non) - Sarah Adibah

## Domain Proyek

*Domain proyek ini membahas permasalahan dalam bidang kesehatan, khususnya mengenai prediksi status diabetes berdasarkan data klinis dan demografis pasien.*
 
Proyek ini bertujuan untuk memprediksi apakah seseorang menderita diabetes atau tidak dengan menggunakan fitur-fitur seperti usia, riwayat merokok, tekanan darah, indeks massa tubuh (_BMI_), kadar HbA1c, dan kadar glukosa darah. Data yang digunakan mencakup beragam ras, lokasi, dan kondisi kesehatan pasien, sehingga dapat digunakan untuk membangun model klasifikasi yang dapat membantu deteksi dini diabetes secara lebih akurat.

<img src="https://cdn.rri.co.id/infografis/images/1676072537-prevalensi_diabetes_di_indonesia.jpg" alt="Data Diabetes" title="Data Diabetes" width="100%">

Diabetes merupakan salah satu penyakit kronis yang menjadi perhatian global karena jumlah penderitanya yang terus meningkat dari tahun ke tahun. Data dari International Diabetes Federation (IDF) menunjukkan bahwa pada tahun 2021, terdapat sekitar 537 juta penderita diabetes di seluruh dunia. Angka ini diproyeksikan akan meningkat menjadi 643 juta pada tahun 2030 dan mencapai 783 juta pada tahun 2045. [[1]](http://sehatnegeriku.kemkes.go.id/baca/blog/20240110/5344736/saatnya-mengatur-si-manis/)

Secara global, pada tahun 2024, diperkirakan terdapat sekitar 589 juta orang dewasa (usia 20–79 tahun) yang hidup dengan diabetes, dan jumlah ini diprediksi akan mencapai 853 juta pada tahun 2050. Fakta lainnya yang mencengangkan adalah bahwa lebih dari 4 dari 5 orang dewasa (sekitar 81%) yang hidup dengan diabetes tinggal di negara-negara berpenghasilan rendah dan menengah, termasuk Indonesia. [[2]](https://diabetesatlas.org/data-by-location/global/)

Di Indonesia sendiri, pada tahun 2021, tercatat sekitar 19,5 juta penderita diabetes. Angka ini diprediksi meningkat menjadi 28,6 juta pada tahun 2045, menempatkan Indonesia sebagai negara dengan jumlah penderita diabetes kelima terbanyak di dunia . Persoalan ini menjadi perhatian serius dari Kementerian Kesehatan karena diabetes melitus merupakan induk dari berbagai penyakit kronis lainnya yang dapat menurunkan kualitas hidup secara signifikan.

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
