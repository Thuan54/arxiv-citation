# Phân tích dữ liệu trên file `bibtex.jsonl`


## Thông tin một bản ghi
Một bản ghi được lưu dưới định dạng `JSON` và có cấu trúc như bên dưới
```JSON
{
    "arxiv_id": str,
    "citation_key": str,
    "entry_type": str | null,
    "title": str | null,
    "authors": [str] | null,
    "year": int | null,
    "doi": str | null,
    "journal": str | null,
    "url": str | null
}
```


*Giải thích ý nghĩa các trường dữ liệu*
| Tên trường (`Field Name`) | Kiểu dữ liệu (`Data Type`) | Giá trị mẫu (`Example Value`) | Ý nghĩa & Vai trò trong Pipeline |
| --- | --- | --- | --- |
| **`arxiv_id`** | `string` (non-null)| `"2408.16061"` | Mã định danh arXiv gốc của bài báo **đang trích dẫn** (citing publication) chứa mục tài liệu tham khảo này. Trường này dùng để nhóm các ứng viên, đảm bảo việc ghép nối (matching) ở Milestone 3 chỉ thực hiện trong phạm vi nội bộ của từng bài báo.|
| **`citation_key`** | `string` (non-null)| `"aanaes2016large"` | Khóa trích dẫn đại diện duy nhất cho mục BibTeX sau khi đã khử trùng lặp (deduplication) và chuẩn hóa lại các lệnh `\cite{}` xuyên suốt các phiên bản của bài báo. Đây cũng là mã định danh của từng dòng dự đoán trong bảng `predictions` ở Milestone 3.|
| **`entry_type`** | `string` (dictionary)| `"article"` | Phân loại ấn phẩm theo chuẩn BibTeX (ví dụ: `article`, `inproceedings`, `misc`), được suy ra từ cấu trúc của lệnh `\bibitem` trong mã nguồn LaTeX.|
| **`title`** | `string`| `"Large-scale data for multiple-view stereopsis"` | Tiêu đề của tài liệu được trích dẫn sau khi đã làm sạch mã định dạng LaTeX và hợp nhất (union) thông tin từ các phiên bản, dùng để tính độ tương đồng văn bản với `ref_title` ở Milestone 3.|
| **`authors`** | `list of strings`| `["Henrik Aanæs", "Rasmus Ramsbøl Jensen", "George Vogiatzis", "Engin Tola", "Anders Bjorholm Dahl"]` | Danh sách tác giả của tài liệu được trích dẫn, dùng để trích xuất đặc trưng trùng khớp tên tác giả (so sánh với `ref_authors`) trong giai đoạn Feature Engineering.|
| **`year`** | `int` (nullable)| `2016` | Năm xuất bản của tài liệu được trích dẫn, dùng để tính độ chênh lệch thời gian với `ref_submission_date` nhằm hỗ trợ mô hình học máy phân loại.|
| **`doi`** | `string`| `null` | Mã định danh đối tượng số (DOI) trích xuất được từ `\bibitem` (bản ghi mẫu này không có DOI nên nhận giá trị `null`).|
| **`journal`** | `string`| `"IJCV"` | Tên tạp chí hoặc kỷ yếu hội nghị khoa học nơi tài liệu được công bố (ở đây là viết tắt của *International Journal of Computer Vision*).|
| **`url`** | `string`| `null` | Đường dẫn URL hoặc liên kết arXiv đi kèm trong mục trích dẫn `\bibitem` (mang giá trị `null` do bản ghi này không chứa liên kết trực tiếp).|