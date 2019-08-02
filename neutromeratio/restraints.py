from simtk import unit

def flatt_bottom_position_restraint(r:float):
    k = 5
    if r <= 0.8:
        e = k * (0.8 - r)**2
    elif r >= 1.2:
        e = k * (r - 1.2)**2
    else:
        e = 0.0
    return e * unit.kilojoule_per_mole