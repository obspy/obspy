struct header
{
	int d_year;
	int d_mon;
	int d_day;
	int t_hour;
	int t_min;
	float t_sec;
	char station[6];
	char channel[4];
	char auxid[5];
	char datatype[4];
	int n_samps;
	float samp_rate;
	float calib;
	float calper;
	char instype[7];
	float hang;
	float vang;
};
