switch dataset
	case 'a'
		dsname = 'alphanumeric', load 'dtset/alphanumeric';
	case 'm'
		dsname = 'MNIST', load 'dtset/MNIST';
	case 'C'
		dsname = 'corel_cedd_10'; pc=35; load 'dtset/corel_cedd_10';
	case 'x'
		regress=1, dsname = 'xydata'; pc=30; load 'dtset/xydataS';
	case 'o'
		dsname = 'coil'; load 'dtset/coil';
	case 'O'
		dsname = 'coil2'; load 'dtset/coil2';
	case 'u'
		dsname = 'USPS', load 'dtset/USPS';
	case 't'
		dsname = 'text', load 'dtset/text', distype=2;
	case 'd'
		dsname = 'digit1', load 'dtset/digit1';
end