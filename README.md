# pairwise-hotupdate-vm-atlas

## 领域

游戏、图形与多媒体

## 主题

热更新版本切换

## 说明

闭包代际、校验门控与回滚清缓存。

初始状态下 `gameplay/hotupdate_vm_atlas.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
