utc_nuclear_test_2013 = UTC('2013-02-12T02:57:51')
ccs = correlate_stream_template(
    stream, template, template_time=utc_nuclear_test_2013)
detections = similarity_detector(ccs, 0.3, 10, 10)
print('number of detections:', len(detections))
print('first detection:', str(detections[0]))
