# Phân tích dữ liệu trong file `references_table.jsonl`
## Một số thuộc tính tổng quan dữ liệu thu được

## Thông tin của một bản ghi
Mỗi bản ghi được lưu trữ ở dạng **JSON** và có cấu trúc như sau
```JSON
{
    "arxiv_id": str, 
    "ref_arxiv_id": str,
    "ref_title": str | null,
    "ref_authors": [str] | null,
    "ref_submission_date": str | null, 
    "ref_s2_id": str | null
}
```

*Giải thích ý nghĩa của các trường dữ liệu*
| Tên trường | Kiểu dữ liệu | Ví dụ | Ý nghĩa và vai trò|
| --- | --- | --- | --- |
| **`arxiv_id`** | `string` (non-null) | `"2408.16045"` | ID của một Arxiv publication. Được sử dụng để xác định các references được sử dụng trong bài báo|
| **`ref_arxiv_id`** | `string` (non-null)| `"0905.4720"` | ID của Arxiv publication được trích dẫn bởi publication có ID **arxiv_id**. Chỉ những publication có Arxiv ID mới được giữ trong bảng này, sử dụng như mục tiêu dự đoán trong milestone 3.|
| **`ref_title`** | `string`| `"String Axiverse"` | Tiêu đề bài báo thu được từ Semantic Scholar API, được sử dụng để tính toán similarity so với những gì thu được từ Bibtex titles.|
| **`ref_authors`** | `list of strings`| `["A. Arvanitaki", "S. Dimopoulos", "S. Dubovsky", "N. Kaloper", "J. March-Russell"]` | Danh sách các tác giả của publication, được sử dụng để so sánh với danh sách tác giẩ trên BibTexThe thông qua **author-overlap engineer** và **matching lastname**.|
| **`ref_submission_date`** | `string`| `"2009-05-28"` | Ngày nộp của publication, sử dụng để so sánh với trường `year` trên BibText.|
| **`ref_s2_id`** | `string`|`"a740f3098064b80472fef5d187812764ddd7da76"` | Mã định danh gồm 40 ký tự của Semantic Scholar(`paperId`) cho mục đích trích dẫn.|

