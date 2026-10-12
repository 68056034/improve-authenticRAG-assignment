import ast
import hashlib
import textwrap
from pathlib import Path

baseline_path = Path('onlysearchAuthenticRAGbak.py')
target_path = Path('onlysearchAuthenticRAG.py')
baseline_bytes = baseline_path.read_bytes()
baseline = baseline_bytes.decode('utf-8-sig')
previous = target_path.read_text(encoding='utf-8-sig')
previous_cls = next(n for n in ast.parse(previous).body if isinstance(n, ast.ClassDef))
rerank_fn = next(n for n in previous_cls.body if isinstance(n, ast.FunctionDef) and n.name == 'rerank')
rerank_code = '\n'.join(previous.splitlines()[rerank_fn.lineno - 1:rerank_fn.end_lineno])
rerank_code = rerank_code.replace(
    'Reranker is disabled; set RAG_USE_RERANKER=1 before starting',
    'Reranker has not been initialized',
)

updated = baseline.replace(
    'from sentence_transformers import SentenceTransformer',
    'from sentence_transformers import SentenceTransformer, CrossEncoder',
    1,
)
anchor = '        self.embedding_dim = self.embed_model.get_sentence_embedding_dimension()'
assert updated.count(anchor) == 1
updated = updated.replace(anchor, anchor + '''

        # เพิ่มเฉพาะ reranking: โหลดครั้งเดียวและใช้ซ้ำทุกคำถาม
        self.reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            max_length=512,
        )''', 1)

anchor = '    def hybrid_search(self, query, k=5, rrf_k=60):'
assert updated.count(anchor) == 1
updated = updated.replace(anchor, rerank_code + '\n\n' + anchor, 1)
start = updated.index(anchor)
end = updated.index('    def truncate_context(', start)
hybrid = updated[start:end]
assert hybrid.count('        return fused_results[:k]') == 1
hybrid = hybrid.replace('        return fused_results[:k]',
'''        # ให้คะแนนผู้สมัครชุดเดิมทั้งหมดก่อนเลือก Top-k
        return self.rerank(query, fused_results, top_k=k)''', 1)
updated = updated[:start] + hybrid + updated[end:]

# Keep result files separate so the reranked run cannot overwrite the baseline.
assert updated.count('"authentic_rag_search_resultsbak.json"') == 1
updated = updated.replace('"authentic_rag_search_resultsbak.json"',
                          '"authentic_rag_search_results.json"', 1)

before_tree = ast.parse(baseline)
after_tree = ast.parse(updated)
before_cls = next(n for n in before_tree.body if isinstance(n, ast.ClassDef))
after_cls = next(n for n in after_tree.body if isinstance(n, ast.ClassDef))
before_methods = {n.name: n for n in before_cls.body if isinstance(n, ast.FunctionDef)}
after_methods = {n.name: n for n in after_cls.body if isinstance(n, ast.FunctionDef)}
assert set(after_methods) == set(before_methods) | {'rerank'}
for name, method in before_methods.items():
    if name not in {'__init__', 'hybrid_search'}:
        assert ast.dump(method) == ast.dump(after_methods[name]), name

# Removing the one reranker initialization restores the exact baseline constructor.
after_init = after_methods['__init__']
after_init.body = [n for n in after_init.body if not (
    isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Attribute) and t.attr == 'reranker' for t in n.targets
    )
)]
assert ast.dump(after_init) == ast.dump(before_methods['__init__'])
assert not any(word in updated for word in ('AutoTokenizer', '_fit_answer_context',
    'RAG_MAX_PROMPT_TOKENS', 'RAG_USE_RERANKER', 'prompt_tokens_sent_local'))

Path('work/onlysearchAuthenticRAG.previous.py').write_text(previous, encoding='utf-8')
target_path.write_text(updated, encoding='utf-8')
assert baseline_path.read_bytes() == baseline_bytes
print('Updated onlysearchAuthenticRAG.py from baseline + reranking only.')
print('Verified syntax and identical token/context, generation, retrieval and JSON schema methods.')
print('Baseline unchanged:', hashlib.sha256(baseline_bytes).hexdigest())
print('Result filenames remain separate; no inference calls or JSON changes.')
