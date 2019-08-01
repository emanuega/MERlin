def test_get_analysis_tasks(simple_data, simple_task):
    assert len(simple_data.get_analysis_tasks()) == 0
    simple_task.save()
    assert len(simple_data.get_analysis_tasks()) == 1
    assert simple_data.get_analysis_tasks()[0]\
           == simple_task.get_analysis_name()
