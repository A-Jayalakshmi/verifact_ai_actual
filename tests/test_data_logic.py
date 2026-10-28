from verifact.core import clean,source_url,explain

def test_clean():
    assert clean(' -LRB- hello -RRB- ')=='( hello )'
def test_url():
    assert 'Albert_Einstein' in source_url('Albert_Einstein')
def test_no_evidence():
    assert 'No relevant evidence' in explain('test',{'label':'NOT ENOUGH INFO'})
