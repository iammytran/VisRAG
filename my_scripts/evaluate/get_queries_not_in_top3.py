def main(k=3, qrels_dict=None, output_file="debug/not_in_top3.txt"):
    top_k_results = {}
    all_queries = set()
    matched_queries = {}
    folder_result = "/Users/mytnguyen/Documents/VisRAG/checkpoints/eval-2026-08-09-111830-maxq-512-maxp-2048-bsz-4-pooling-wmean-attention-causal-gpus-per-node-4/InfoVQA"
    for i in range (4):
        file_format = f"test.{i}.trec"
        with open(f"{folder_result}/{file_format}", 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue

                qid = parts[0]
                doc_id = parts[2]
                rank = int(parts[3])
                score = parts[4] if len(parts) > 4 else "N/A"

                all_queries.add(qid)

                if rank <= k:
                    if qid not in top_k_results:
                        top_k_results[qid] = []
                    top_k_results[qid].append((file_format, doc_id, rank, score))


        failed_queries = {}

        # 2. Lọc ra các query bị hỏng (Không có đáp án đúng trong Top K)
        for qid in sorted(all_queries):
            if qrels_dict and qid in qrels_dict:
                relevant_docs = qrels_dict[qid]
            else:
                # Mặc định suy ra từ tên query: '36966.jpeg-1' -> '36966.jpeg'
                relevant_docs = {qid.rsplit("-", 1)[0]}

            retrieved_items = top_k_results.get(qid, [])
            retrieved_doc_ids = {item[1] for item in retrieved_items}
            related_trec_file = {item[0] for item in retrieved_items}

            # Nếu không có đáp án đúng nào nằm trong danh sách Top K đã lấy ra
            if not (relevant_docs & retrieved_doc_ids):
                # Sắp xếp các kết quả theo đúng thứ tự Rank 1 -> K
                retrieved_items.sort(key=lambda x: x[1])
                failed_queries[qid] = {
                    "relevant_docs": list(relevant_docs),
                    "top_k": retrieved_items,
                    "related_trec_file": related_trec_file
                }

    # 3. Ghi dữ liệu ra file text
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(
            f"=== DANH SÁCH {len(failed_queries)} QUERY KHÔNG CÓ ĐÁP ÁN ĐÚNG TRONG TOP {k} ===\n\n"
        )

        for qid, info in failed_queries.items():
            f.write(f"Query ID   : {qid} - {info['related_trec_file']}\n")
            f.write(
                f"Đáp án đúng: {', '.join(info['relevant_docs'])}\n"
            )
            f.write(f"Top {k} kết quả sai mô hình trả về:\n")

            for _, doc_id, rank, score in info["top_k"]:
                f.write(f"   [Rank {rank}] {doc_id:<15} (Score: {score})\n")

            f.write("-" * 55 + "\n")

    print(
        f"✅ Đã hoàn tất! Có {len(failed_queries)} query sai. Chi tiết đã lưu tại '{output_file}'."
    )

if __name__ == "__main__":
    # results = main(k=3)

    # print(f"--- CÁC CÂU HỎI CÓ ĐÁP ÁN ĐÚNG TRONG TOP 3 ({len(results)}) ---")
    # for qid, info in results.items():
    #     print(
    #         f"Query: {qid:<15} | Đáp án: {info['correct_doc']:<12} | Rank: {info['rank']}"
    #     )
    not_top_3 = main(k=3)