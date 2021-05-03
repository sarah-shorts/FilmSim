#PhysCat end of year project
##Simulating Camera Film
#functions page
###hi aditya!!

@vectorize
def did_it_couple(f_spectral_val, intensity):
    if random.random() < f_spectral_val*intensity/65535:
        return True
    else: 
        return False
#this is to simulate the physical boolean nature of whether or not each crystal will react to the light and bond with a coupler to form color  
#vectorizing it sped the simulation up considerably because this is the most computed funtion in the simulation computed 2.4*10^9 times per image

@jit
def compute_pixels(film, image, y_specval, c_specval, m_specval):
    for row in range(512):
            for col in range(512):
                film[row][col][0] += np.sum(np.array([did_it_couple(c_specval, image[row][col]) for _ in range(100)]))
                film[row][col][1] += np.sum(np.array([did_it_couple(m_specval, image[row][col]) for _ in range(100)]))
                film[row][col][2] += np.sum(np.array([did_it_couple(y_specval, image[row][col]) for _ in range(100)]))
    return film
#this is only computed 31 times but it is a very long part of the simulation to compute and is easily "jit"-ed so that it takes much less time
#this part of the function serves to calculate all the colors


def take_a_pic(image_folder, f_cyan, f_magenta, f_yellow):
    IMG = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
            im = np.array(imageio.imread(image_folder+"/"+filename))
            IMG.append(im)
    IMG = np.array(IMG)
    print(IMG.shape)
    film = np.zeros([512,512,3])
    for index, image in enumerate(IMG):
        wavelength = 400+index*10
        y_specval = f_yellow(wavelength)
        c_specval = f_cyan(wavelength)
        m_specval = f_magenta(wavelength)
        film = compute_pixels(film, image, y_specval, c_specval, m_specval)
        norm = np.max(film)/100
        film[::][::][0] += (1-film[::][::][0]/(100))
        film[::][::][1] += (1-film[::][::][1]/(100))
        film[::][::][2] += (1-film[::][::][2]/(100))
    return np.array(film)/(len(IMG)*norm)
#this adds up the image for each of the different wavelengths given in the specific file format

#[ok I know this seems simple but I'm dumb and it took me a very long time :( ]
